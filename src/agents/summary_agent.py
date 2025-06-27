"""
Summary Agent for generating different types of summaries.
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class SummaryAgent(BaseAgent):
    """Agent specialized in generating various types of summaries."""

    def __init__(
        self,
        name: str = "summary_agent",
        azure_endpoint: str = "",
        azure_api_key: str = "",
        azure_deployment: str = "",
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.3
    ) -> None:
        """
        Initialize Summary Agent.
        
        Args:
            name: Agent name
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key
            azure_deployment: Azure OpenAI deployment name
            api_version: API version
            temperature: Model temperature for generation
        """
        self._name = name
        
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
            logger.warning("Azure OpenAI credentials not provided. Summary agent will have limited functionality.")

        # Summary type templates
        self.summary_templates = {
            "executive": self._create_executive_template(),
            "technical": self._create_technical_template(),
            "narrative": self._create_narrative_template(),
            "bullet_points": self._create_bullet_template(),
            "abstract": self._create_abstract_template()
        }

    @property
    def name(self) -> str:
        return self._name

    def can_handle(self, request: str) -> bool:
        """Determine if this agent can handle summary requests."""
        summary_keywords = [
            'summarize', 'summary', 'sum up', 'brief', 'overview',
            'abstract', 'executive summary', 'key points', 'highlights',
            'tldr', 'tl;dr', 'gist', 'synopsis', 'digest'
        ]
        return any(keyword in request.lower() for keyword in summary_keywords)

    def process(self, request: str) -> dict:
        """Process the request using the configured strategy."""
        return self.retrieve_and_generate(request)

    def retrieve_and_generate(self, request: str) -> dict:
        """Generate summaries based on the request."""
        if not self.llm:
            return {
                'answer': "Summary agent is not properly configured. Please provide Azure OpenAI credentials.",
                'summary_type': 'error',
                'confidence': 0.0
            }

        # Detect summary type from request
        summary_type = self._detect_summary_type(request)
        
        # Extract content to summarize
        content = self._extract_content_from_request(request)
        
        if not content:
            return {
                'answer': "I need content to summarize. Please provide text or specify what you'd like me to summarize.",
                'summary_type': summary_type,
                'confidence': 0.0
            }

        try:
            # Generate summary using appropriate template
            template = self.summary_templates.get(summary_type, self.summary_templates['narrative'])
            prompt = template.format_messages(content=content, request=request)
            
            response = self.llm.invoke(prompt)
            summary_text = response.content.strip()

            return {
                'answer': summary_text,
                'summary_type': summary_type,
                'confidence': 0.9,
                'word_count': len(summary_text.split()),
                'original_length': len(content.split()),
                'compression_ratio': len(summary_text.split()) / len(content.split()) if content else 0
            }

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                'answer': f"Error generating summary: {str(e)}",
                'summary_type': summary_type,
                'confidence': 0.0
            }

    def summarize(self, request: str) -> dict:
        """Generate a summary for the given request."""
        return self.retrieve_and_generate(request)

    def _detect_summary_type(self, request: str) -> str:
        """Detect the type of summary requested."""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['executive', 'board', 'management', 'high-level']):
            return 'executive'
        elif any(word in request_lower for word in ['technical', 'detailed', 'deep', 'analysis']):
            return 'technical'
        elif any(word in request_lower for word in ['bullet', 'points', 'list', 'key points']):
            return 'bullet_points'
        elif any(word in request_lower for word in ['abstract', 'academic', 'research']):
            return 'abstract'
        else:
            return 'narrative'

    def _extract_content_from_request(self, request: str) -> str:
        """Extract content to summarize from the request."""
        # Look for content after keywords like "summarize this:", "summary of:", etc.
        patterns = [
            'summarize this:', 'summarize the following:', 'summary of:',
            'sum up:', 'brief on:', 'overview of:', 'abstract of:'
        ]
        
        request_lower = request.lower()
        for pattern in patterns:
            if pattern in request_lower:
                content_start = request_lower.find(pattern) + len(pattern)
                return request[content_start:].strip()
        
        # If no explicit pattern, assume the entire request contains content
        # Remove the summary instruction part
        summary_words = ['summarize', 'summary', 'sum up', 'brief', 'overview', 'abstract']
        words = request.split()
        
        # Find first non-summary word
        content_start = 0
        for i, word in enumerate(words):
            if word.lower() not in summary_words:
                content_start = i
                break
        
        if content_start > 0:
            return ' '.join(words[content_start:])
        
        return request

    def _create_executive_template(self) -> ChatPromptTemplate:
        """Template for executive summaries."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert executive summary writer. Create concise, high-level summaries 
            that focus on key insights, strategic implications, and actionable outcomes. Your summaries should be:
            - Brief and to the point (2-3 paragraphs maximum)
            - Focused on business impact and strategic value
            - Written for senior leadership
            - Free of technical jargon unless essential"""),
            ("human", "Create an executive summary of the following content:\n\n{content}\n\nUser request: {request}")
        ])

    def _create_technical_template(self) -> ChatPromptTemplate:
        """Template for technical summaries."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a technical documentation expert. Create detailed technical summaries that:
            - Preserve important technical details and specifications
            - Include methodology and implementation details
            - Maintain technical accuracy
            - Organize information logically
            - Include relevant metrics and data points"""),
            ("human", "Create a technical summary of the following content:\n\n{content}\n\nUser request: {request}")
        ])

    def _create_narrative_template(self) -> ChatPromptTemplate:
        """Template for narrative summaries."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a skilled writer who creates engaging narrative summaries. Your summaries should:
            - Tell a coherent story with clear flow
            - Maintain the essential message and tone
            - Be accessible to a general audience
            - Include context and background where helpful
            - Be well-structured with smooth transitions"""),
            ("human", "Create a narrative summary of the following content:\n\n{content}\n\nUser request: {request}")
        ])

    def _create_bullet_template(self) -> ChatPromptTemplate:
        """Template for bullet-point summaries."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating clear, organized bullet-point summaries. Your summaries should:
            - Use clear, concise bullet points
            - Organize information hierarchically when appropriate
            - Highlight the most important points first
            - Use consistent formatting
            - Be scannable and easy to read"""),
            ("human", "Create a bullet-point summary of the following content:\n\n{content}\n\nUser request: {request}")
        ])

    def _create_abstract_template(self) -> ChatPromptTemplate:
        """Template for academic abstracts."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an academic writing expert who creates scholarly abstracts. Your abstracts should:
            - Follow academic writing conventions
            - Include objective, methodology, results, and conclusions
            - Be precise and formal in tone
            - Avoid subjective language
            - Be suitable for academic or research contexts"""),
            ("human", "Create an academic abstract of the following content:\n\n{content}\n\nUser request: {request}")
        ])

    def generate_summary_by_type(self, content: str, summary_type: str) -> dict:
        """Generate a specific type of summary for given content."""
        if summary_type not in self.summary_templates:
            return {
                'answer': f"Unknown summary type: {summary_type}. Available types: {list(self.summary_templates.keys())}",
                'summary_type': 'error',
                'confidence': 0.0
            }

        try:
            template = self.summary_templates[summary_type]
            prompt = template.format_messages(content=content, request=f"Generate {summary_type} summary")
            
            response = self.llm.invoke(prompt)
            summary_text = response.content.strip()

            return {
                'answer': summary_text,
                'summary_type': summary_type,
                'confidence': 0.9,
                'word_count': len(summary_text.split()),
                'original_length': len(content.split()),
                'compression_ratio': len(summary_text.split()) / len(content.split()) if content else 0
            }

        except Exception as e:
            logger.error(f"Error generating {summary_type} summary: {e}")
            return {
                'answer': f"Error generating {summary_type} summary: {str(e)}",
                'summary_type': summary_type,
                'confidence': 0.0
            } 