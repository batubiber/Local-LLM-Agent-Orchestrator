"""
Agent Template for creating new agents.

This template provides a standardized structure for creating new agents
that follow the BaseAgent protocol and integrate seamlessly with the orchestrator.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentTemplate(BaseAgent):
    """
    Template for creating new agents.
    
    Replace 'AgentTemplate' with your actual agent name (e.g., 'WeatherAgent').
    Implement the required methods and customize the functionality as needed.
    """

    def __init__(
        self,
        name: str = "template_agent",
        azure_endpoint: str = "",
        azure_api_key: str = "",
        azure_deployment: str = "",
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.5,
        # Add your custom parameters here
        custom_param: Optional[str] = None
    ) -> None:
        """
        Initialize your agent.
        
        Args:
            name: Agent name (should be unique)
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key
            azure_deployment: Azure OpenAI deployment name
            api_version: API version
            temperature: Model temperature for generation
            custom_param: Example of a custom parameter
        """
        self._name = name
        self.custom_param = custom_param
        
        # Initialize Azure OpenAI client if credentials are provided
        if azure_endpoint and azure_api_key and azure_deployment:
            self.llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                azure_deployment=azure_deployment,
                api_version=api_version,
                temperature=temperature
            )
        else:
            self.llm = None
            logger.warning(f"{name} agent: Azure OpenAI credentials not provided. Limited functionality available.")

        # Initialize any agent-specific components
        self._initialize_agent_components()

    @property
    def name(self) -> str:
        """Return the agent's unique name."""
        return self._name

    def can_handle(self, request: str) -> bool:
        """
        Determine if this agent can handle the given request.
        
        Args:
            request: The user's request/query
            
        Returns:
            bool: True if this agent can handle the request
            
        Example implementation:
        """
        # Define keywords that indicate this agent should handle the request
        keywords = [
            'your', 'agent', 'specific', 'keywords', 'here'
            # Example: 'weather', 'forecast', 'temperature', 'rain'
        ]
        
        request_lower = request.lower()
        return any(keyword in request_lower for keyword in keywords)

    def process(self, request: str) -> dict:
        """
        Process the request using the configured strategy.
        
        This method typically delegates to retrieve_and_generate,
        but can be customized for specific processing needs.
        """
        return self.retrieve_and_generate(request)

    def retrieve_and_generate(self, request: str) -> dict:
        """
        Perform the main processing logic for the agent.
        
        Args:
            request: The user's request/query
            
        Returns:
            dict: Response containing the agent's answer and metadata
        """
        if not self.llm:
            return {
                'answer': f"{self.name} agent is not properly configured. Please provide Azure OpenAI credentials.",
                'confidence': 0.0,
                'agent': self.name
            }

        try:
            # Your main processing logic goes here
            
            # Example: Create a prompt template
            prompt_template = self._create_processing_template()
            
            # Prepare the context for the prompt
            context = self._prepare_context(request)
            
            # Generate the prompt
            prompt = prompt_template.format_messages(**context)
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            result = response.content.strip()

            # Process and format the result
            return {
                'answer': result,
                'confidence': 0.9,
                'agent': self.name,
                'request_type': self._detect_request_type(request),
                # Add any additional metadata
                'custom_metadata': self._get_custom_metadata(request, result)
            }

        except Exception as e:
            logger.error(f"Error in {self.name} agent: {e}")
            return {
                'answer': f"Error processing request: {str(e)}",
                'confidence': 0.0,
                'agent': self.name,
                'error': str(e)
            }

    def summarize(self, request: str) -> dict:
        """
        Generate a summary for the given request.
        
        This method can be customized for agent-specific summarization,
        or it can delegate to retrieve_and_generate.
        """
        # Option 1: Create a summary-specific implementation
        summary_request = f"Summarize: {request}"
        return self.retrieve_and_generate(summary_request)
        
        # Option 2: Delegate to main processing
        # return self.retrieve_and_generate(request)

    def _initialize_agent_components(self) -> None:
        """
        Initialize agent-specific components.
        
        Override this method to set up any additional resources,
        databases, APIs, or other components your agent needs.
        """
        # Example: Initialize external APIs, databases, etc.
        pass

    def _create_processing_template(self) -> ChatPromptTemplate:
        """
        Create the main prompt template for processing requests.
        
        Returns:
            ChatPromptTemplate: The template for generating responses
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant specialized in [YOUR DOMAIN].
            Your role is to [DESCRIBE YOUR AGENT'S PURPOSE].
            
            Guidelines:
            - Be accurate and helpful
            - Provide specific and actionable information
            - Cite sources when appropriate
            - Admit when you don't know something
            """),
            ("human", "{request}")
        ])

    def _prepare_context(self, request: str) -> Dict[str, Any]:
        """
        Prepare context variables for the prompt template.
        
        Args:
            request: The user's request
            
        Returns:
            dict: Context variables for the prompt
        """
        return {
            'request': request,
            'agent_name': self.name,
            # Add any additional context your agent needs
        }

    def _detect_request_type(self, request: str) -> str:
        """
        Detect the type of request to customize processing.
        
        Args:
            request: The user's request
            
        Returns:
            str: The type of request (e.g., 'query', 'analysis', 'generation')
        """
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['analyze', 'analysis', 'examine']):
            return 'analysis'
        elif any(word in request_lower for word in ['generate', 'create', 'make']):
            return 'generation'
        elif any(word in request_lower for word in ['summarize', 'summary']):
            return 'summary'
        else:
            return 'query'

    def _get_custom_metadata(self, request: str, result: str) -> Dict[str, Any]:
        """
        Generate custom metadata for the response.
        
        Args:
            request: The original request
            result: The generated result
            
        Returns:
            dict: Custom metadata specific to your agent
        """
        return {
            'request_length': len(request.split()),
            'response_length': len(result.split()),
            'processing_mode': 'llm' if self.llm else 'fallback',
            # Add any agent-specific metadata
        }

    # Add any additional methods your agent needs
    def custom_method(self, parameter: str) -> Any:
        """
        Example of a custom method specific to your agent.
        
        Args:
            parameter: Custom parameter
            
        Returns:
            Any: Custom result
        """
        # Implement your custom functionality
        pass


# Example of how to use the template:

class ExampleWeatherAgent(AgentTemplate):
    """
    Example agent for weather information.
    This shows how to customize the template for a specific domain.
    """

    def __init__(self, **kwargs):
        # Set default name if not provided
        kwargs.setdefault('name', 'weather_agent')
        super().__init__(**kwargs)

    def can_handle(self, request: str) -> bool:
        """Check if request is weather-related."""
        weather_keywords = [
            'weather', 'temperature', 'forecast', 'rain', 'snow',
            'sunny', 'cloudy', 'wind', 'humidity', 'climate'
        ]
        request_lower = request.lower()
        return any(keyword in request_lower for keyword in weather_keywords)

    def _create_processing_template(self) -> ChatPromptTemplate:
        """Create weather-specific prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a weather information assistant. Provide accurate, 
            helpful weather information and forecasts. If you need real-time data,
            explain what information would be needed and suggest reliable sources."""),
            ("human", "Weather request: {request}")
        ])

    def _get_custom_metadata(self, request: str, result: str) -> Dict[str, Any]:
        """Add weather-specific metadata."""
        metadata = super()._get_custom_metadata(request, result)
        metadata.update({
            'weather_request': True,
            'location_mentioned': self._extract_location(request)
        })
        return metadata

    def _extract_location(self, request: str) -> Optional[str]:
        """Extract location from weather request."""
        # Simple implementation - you could use NER for better results
        import re
        location_pattern = r'\b(?:in|at|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        match = re.search(location_pattern, request)
        return match.group(1) if match else None


# Usage instructions:
"""
To create a new agent:

1. Copy this template file and rename it (e.g., 'weather_agent.py')
2. Replace 'AgentTemplate' with your agent's name (e.g., 'WeatherAgent')
3. Implement the required methods:
   - can_handle(): Define what requests this agent handles
   - retrieve_and_generate(): Implement main processing logic
4. Customize optional methods as needed:
   - _create_processing_template(): Define agent-specific prompts
   - _initialize_agent_components(): Set up external resources
   - _get_custom_metadata(): Add agent-specific response metadata
5. Register your agent in the orchestrator
6. Add any necessary dependencies to requirements.txt

Example registration in orchestrator:
```python
from src.agents.your_agent import YourAgent

# In create_orchestrator function:
your_agent = YourAgent(
    name="your_agent",
    azure_endpoint=azure_endpoint,
    azure_api_key=azure_api_key,
    azure_deployment=azure_deployment
)
agents.append(your_agent)
strategies["your_agent"] = RAGStrategy()
```
""" 