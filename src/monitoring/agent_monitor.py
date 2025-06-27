"""
Agent monitoring system for tracking performance and health.
"""
from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Metrics for individual agent performance."""
    agent_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_messages: deque = field(default_factory=lambda: deque(maxlen=50))
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    def update_response_time(self, response_time: float) -> None:
        """Update response time metrics."""
        self.response_times.append(response_time)
        if self.response_times:
            self.average_response_time = sum(self.response_times) / len(self.response_times)
    
    def add_confidence_score(self, score: float) -> None:
        """Add confidence score to metrics."""
        if 0.0 <= score <= 1.0:
            self.confidence_scores.append(score)
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


@dataclass
class RequestLog:
    """Log entry for individual requests."""
    timestamp: datetime
    agent_name: str
    request: str
    response: Dict[str, Any]
    response_time: float
    success: bool
    error_message: Optional[str] = None
    confidence: Optional[float] = None


class AgentMonitor:
    """Monitor and track agent performance."""
    
    def __init__(self, max_logs: int = 1000):
        """
        Initialize agent monitor.
        
        Args:
            max_logs: Maximum number of request logs to keep
        """
        self.max_logs = max_logs
        self.metrics: Dict[str, AgentMetrics] = {}
        self.request_logs: deque = deque(maxlen=max_logs)
        self.system_stats = {
            'start_time': datetime.now(),
            'total_requests': 0,
            'total_agents': 0
        }
        self._lock = threading.Lock()
    
    def log_request(
        self,
        agent_name: str,
        request: str,
        response: Dict[str, Any],
        response_time: float,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Log a request and update metrics.
        
        Args:
            agent_name: Name of the agent that processed the request
            request: The original request
            response: The agent's response
            response_time: Time taken to process the request (seconds)
            success: Whether the request was successful
            error_message: Error message if request failed
        """
        with self._lock:
            timestamp = datetime.now()
            
            # Extract confidence score if available
            confidence = None
            if isinstance(response, dict):
                confidence = response.get('confidence')
            
            # Create request log
            log_entry = RequestLog(
                timestamp=timestamp,
                agent_name=agent_name,
                request=request,
                response=response,
                response_time=response_time,
                success=success,
                error_message=error_message,
                confidence=confidence
            )
            
            self.request_logs.append(log_entry)
            
            # Update agent metrics
            if agent_name not in self.metrics:
                self.metrics[agent_name] = AgentMetrics(agent_name=agent_name)
                self.system_stats['total_agents'] += 1
            
            metrics = self.metrics[agent_name]
            metrics.total_requests += 1
            metrics.last_request_time = timestamp
            metrics.update_response_time(response_time)
            
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
                if error_message:
                    metrics.error_messages.append({
                        'timestamp': timestamp,
                        'message': error_message
                    })
            
            if confidence is not None:
                metrics.add_confidence_score(confidence)
            
            self.system_stats['total_requests'] += 1
    
    def get_agent_metrics(self, agent_name: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent."""
        return self.metrics.get(agent_name)
    
    def get_all_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all agents."""
        return self.metrics.copy()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        with self._lock:
            uptime = datetime.now() - self.system_stats['start_time']
            
            stats = self.system_stats.copy()
            stats.update({
                'uptime_seconds': uptime.total_seconds(),
                'uptime_formatted': str(uptime),
                'active_agents': len(self.metrics),
                'average_requests_per_agent': (
                    stats['total_requests'] / len(self.metrics) 
                    if self.metrics else 0
                ),
                'recent_activity': self._get_recent_activity()
            })
            
            return stats
    
    def get_agent_health(self, agent_name: str) -> Dict[str, Any]:
        """
        Get health status for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            dict: Health status information
        """
        metrics = self.metrics.get(agent_name)
        if not metrics:
            return {'status': 'unknown', 'message': 'Agent not found'}
        
        # Determine health status
        health_status = 'healthy'
        health_messages = []
        
        # Check error rate
        if metrics.error_rate > 20:
            health_status = 'degraded'
            health_messages.append(f'High error rate: {metrics.error_rate:.1f}%')
        elif metrics.error_rate > 50:
            health_status = 'unhealthy'
            health_messages.append(f'Very high error rate: {metrics.error_rate:.1f}%')
        
        # Check response time
        if metrics.average_response_time > 10:
            if health_status == 'healthy':
                health_status = 'degraded'
            health_messages.append(f'Slow response time: {metrics.average_response_time:.2f}s')
        elif metrics.average_response_time > 30:
            health_status = 'unhealthy'
            health_messages.append(f'Very slow response time: {metrics.average_response_time:.2f}s')
        
        # Check recent activity
        if metrics.last_request_time:
            time_since_last = datetime.now() - metrics.last_request_time
            if time_since_last > timedelta(hours=1):
                if health_status == 'healthy':
                    health_status = 'idle'
                health_messages.append(f'No recent activity: {time_since_last}')
        
        return {
            'status': health_status,
            'messages': health_messages,
            'metrics': {
                'total_requests': metrics.total_requests,
                'success_rate': metrics.success_rate,
                'error_rate': metrics.error_rate,
                'average_response_time': metrics.average_response_time,
                'average_confidence': metrics.average_confidence,
                'last_request': metrics.last_request_time.isoformat() if metrics.last_request_time else None
            }
        }
    
    def get_recent_logs(self, limit: int = 100, agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent request logs.
        
        Args:
            limit: Maximum number of logs to return
            agent_name: Filter by specific agent (optional)
            
        Returns:
            List of recent request logs
        """
        logs = list(self.request_logs)
        
        if agent_name:
            logs = [log for log in logs if log.agent_name == agent_name]
        
        # Sort by timestamp (most recent first) and limit
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        logs = logs[:limit]
        
        # Convert to serializable format
        serializable_logs = []
        for log in logs:
            log_dict = {
                'timestamp': log.timestamp.isoformat(),
                'agent_name': log.agent_name,
                'request': log.request[:200] + '...' if len(log.request) > 200 else log.request,
                'response_time': log.response_time,
                'success': log.success,
                'confidence': log.confidence,
                'error_message': log.error_message
            }
            serializable_logs.append(log_dict)
        
        return serializable_logs
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export metrics to a JSON file.
        
        Args:
            filepath: Path to save the metrics file
        """
        export_data = {
            'system_stats': self.get_system_stats(),
            'agent_metrics': {},
            'recent_logs': self.get_recent_logs(500)
        }
        
        # Convert metrics to serializable format
        for agent_name, metrics in self.metrics.items():
            export_data['agent_metrics'][agent_name] = {
                'agent_name': metrics.agent_name,
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'success_rate': metrics.success_rate,
                'error_rate': metrics.error_rate,
                'average_response_time': metrics.average_response_time,
                'average_confidence': metrics.average_confidence,
                'last_request_time': metrics.last_request_time.isoformat() if metrics.last_request_time else None,
                'recent_response_times': list(metrics.response_times),
                'recent_confidence_scores': list(metrics.confidence_scores)
            }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
    
    def _get_recent_activity(self) -> Dict[str, int]:
        """Get recent activity statistics."""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        recent_logs = [log for log in self.request_logs if log.timestamp >= last_hour]
        daily_logs = [log for log in self.request_logs if log.timestamp >= last_day]
        
        return {
            'requests_last_hour': len(recent_logs),
            'requests_last_day': len(daily_logs),
            'active_agents_last_hour': len(set(log.agent_name for log in recent_logs)),
            'active_agents_last_day': len(set(log.agent_name for log in daily_logs))
        }
    
    def reset_metrics(self, agent_name: Optional[str] = None) -> None:
        """
        Reset metrics for an agent or all agents.
        
        Args:
            agent_name: Specific agent to reset, or None for all agents
        """
        with self._lock:
            if agent_name:
                if agent_name in self.metrics:
                    del self.metrics[agent_name]
                    logger.info(f"Reset metrics for agent: {agent_name}")
            else:
                self.metrics.clear()
                self.request_logs.clear()
                self.system_stats = {
                    'start_time': datetime.now(),
                    'total_requests': 0,
                    'total_agents': 0
                }
                logger.info("Reset all metrics")


# Global monitor instance
monitor = AgentMonitor()


def log_agent_request(
    agent_name: str,
    request: str,
    response: Dict[str, Any],
    response_time: float,
    success: bool = True,
    error_message: Optional[str] = None
) -> None:
    """
    Convenience function to log agent requests.
    
    Args:
        agent_name: Name of the agent
        request: The request that was processed
        response: The agent's response
        response_time: Time taken to process the request
        success: Whether the request was successful
        error_message: Error message if request failed
    """
    monitor.log_request(
        agent_name=agent_name,
        request=request,
        response=response,
        response_time=response_time,
        success=success,
        error_message=error_message
    )


def get_monitor() -> AgentMonitor:
    """Get the global monitor instance."""
    return monitor 