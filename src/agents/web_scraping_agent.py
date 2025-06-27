"""
Web Scraping Agent for extracting knowledge from web pages and articles.
"""
from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class WebScrapingAgent(BaseAgent):
    """Agent specialized in web scraping and online content analysis."""

    def __init__(
        self,
        name: str = "web_scraping_agent",
        azure_endpoint: str = "",
        azure_api_key: str = "",
        azure_deployment: str = "",
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.2,
        max_content_length: int = 10000,
        timeout: int = 10
    ) -> None:
        """
        Initialize Web Scraping Agent.
        
        Args:
            name: Agent name
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key
            azure_deployment: Azure OpenAI deployment name
            api_version: API version
            temperature: Model temperature for analysis
            max_content_length: Maximum content length to process
            timeout: Request timeout in seconds
        """
        self._name = name
        self.max_content_length = max_content_length
        self.timeout = timeout
        
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
            logger.warning("Azure OpenAI credentials not provided. Web scraping agent will have limited functionality.")

        # Headers for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        # Analysis templates
        self.analysis_templates = {
            "extract": self._create_extract_template(),
            "summarize": self._create_summarize_template(),
            "analyze": self._create_analyze_template(),
            "fact_check": self._create_fact_check_template()
        }

    @property
    def name(self) -> str:
        return self._name

    def can_handle(self, request: str) -> bool:
        """Determine if this agent can handle web-related requests."""
        web_keywords = [
            'website', 'web page', 'url', 'link', 'scrape', 'extract from web',
            'online', 'internet', 'browse', 'fetch from', 'get from url',
            'http', 'https', 'www.', '.com', '.org', '.net', '.edu',
            'article', 'blog post', 'news', 'webpage content'
        ]
        
        # Check for URLs in the request
        url_pattern = r'https?://[^\s]+'
        has_url = bool(re.search(url_pattern, request))
        
        request_lower = request.lower()
        has_keywords = any(keyword in request_lower for keyword in web_keywords)
        
        return has_keywords or has_url

    def process(self, request: str) -> dict:
        """Process the request using the configured strategy."""
        return self.retrieve_and_generate(request)

    def retrieve_and_generate(self, request: str) -> dict:
        """Extract and analyze web content based on the request."""
        # Extract URLs from request
        urls = self._extract_urls(request)
        
        if not urls:
            return {
                'answer': "I need a URL to scrape web content. Please provide a valid web address.",
                'urls_found': 0,
                'confidence': 0.0
            }

        results = []
        for url in urls[:3]:  # Limit to first 3 URLs
            try:
                content_data = self._scrape_url(url)
                if content_data:
                    analysis_result = self._analyze_content(request, content_data)
                    results.append({
                        'url': url,
                        'title': content_data.get('title', 'Unknown'),
                        'analysis': analysis_result,
                        'word_count': content_data.get('word_count', 0)
                    })
                else:
                    results.append({
                        'url': url,
                        'error': 'Failed to scrape content',
                        'analysis': None
                    })
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                results.append({
                    'url': url,
                    'error': str(e),
                    'analysis': None
                })

        # Compile final response
        if results:
            successful_results = [r for r in results if 'analysis' in r and r['analysis']]
            if successful_results:
                # Combine analyses if multiple URLs
                if len(successful_results) == 1:
                    main_analysis = successful_results[0]['analysis']
                else:
                    main_analysis = self._combine_analyses(request, successful_results)
                
                return {
                    'answer': main_analysis,
                    'urls_processed': len(results),
                    'successful_extractions': len(successful_results),
                    'sources': [{'url': r['url'], 'title': r.get('title')} for r in successful_results],
                    'confidence': 0.8
                }

        return {
            'answer': "Unable to extract content from the provided URLs. Please check if the URLs are accessible.",
            'urls_processed': len(results),
            'successful_extractions': 0,
            'confidence': 0.0
        }

    def summarize(self, request: str) -> dict:
        """Generate a summary of web content."""
        return self.retrieve_and_generate(request)

    def _extract_urls(self, request: str) -> List[str]:
        """Extract URLs from the request."""
        url_pattern = r'https?://[^\s<>"\'`,|\\(){}[\]]+[^\s<>"\'`,|\\(){}[\].,;:!?]'
        urls = re.findall(url_pattern, request)
        
        # Clean and validate URLs
        cleaned_urls = []
        for url in urls:
            # Remove trailing punctuation
            url = re.sub(r'[.,;:!?]+$', '', url)
            
            # Basic validation
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                cleaned_urls.append(url)
        
        return cleaned_urls

    def _scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape content from a URL."""
        try:
            logger.info(f"Scraping URL: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "Unknown Title"
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
                element.decompose()
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            if not content.strip():
                logger.warning(f"No content extracted from {url}")
                return None
            
            # Limit content length
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'word_count': len(content.split()),
                'char_count': len(content)
            }
            
        except requests.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from BeautifulSoup object."""
        # Try to find main content containers
        content_selectors = [
            'main', 'article', '[role="main"]', '.main-content', 
            '.content', '.post-content', '.entry-content', '.article-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # Fallback to body if no main content found
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return ""
        
        # Extract text while preserving some structure
        paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        
        content_parts = []
        for para in paragraphs:
            text = para.get_text().strip()
            if text and len(text) > 20:  # Filter out very short text
                content_parts.append(text)
        
        # If no paragraphs found, get all text
        if not content_parts:
            content_parts = [main_content.get_text()]
        
        return '\n\n'.join(content_parts)

    def _analyze_content(self, request: str, content_data: Dict[str, Any]) -> str:
        """Analyze scraped content using LLM."""
        if not self.llm:
            return f"Content extracted from {content_data['title']}: {content_data['content'][:500]}..."
        
        # Detect analysis type
        analysis_type = self._detect_analysis_type(request)
        
        try:
            template = self.analysis_templates.get(analysis_type, self.analysis_templates['extract'])
            prompt = template.format_messages(
                request=request,
                title=content_data['title'],
                url=content_data['url'],
                content=content_data['content']
            )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return f"Error analyzing content: {str(e)}"

    def _detect_analysis_type(self, request: str) -> str:
        """Detect the type of analysis requested."""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['summarize', 'summary', 'sum up', 'brief']):
            return 'summarize'
        elif any(word in request_lower for word in ['analyze', 'analysis', 'examine', 'study']):
            return 'analyze'
        elif any(word in request_lower for word in ['fact check', 'verify', 'validate', 'check']):
            return 'fact_check'
        else:
            return 'extract'

    def _combine_analyses(self, request: str, results: List[Dict]) -> str:
        """Combine analyses from multiple URLs."""
        if not self.llm:
            combined = "Combined content from multiple sources:\n\n"
            for i, result in enumerate(results, 1):
                combined += f"{i}. {result.get('title', 'Unknown')}: {result['analysis'][:200]}...\n\n"
            return combined
        
        try:
            # Create a combined analysis prompt
            sources_text = ""
            for i, result in enumerate(results, 1):
                sources_text += f"Source {i} - {result.get('title', 'Unknown')} ({result['url']}):\n"
                sources_text += f"{result['analysis']}\n\n"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert analyst. Combine and synthesize information from multiple web sources 
                to provide a comprehensive response to the user's request. Identify common themes, contradictions, 
                and provide a balanced analysis citing the sources."""),
                ("human", """User request: {request}

Multiple source analyses:
{sources}

Provide a comprehensive combined analysis that addresses the user's request.""")
            ])
            
            messages = prompt.format_messages(request=request, sources=sources_text)
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error combining analyses: {e}")
            return f"Error combining analyses: {str(e)}"

    def _create_extract_template(self) -> ChatPromptTemplate:
        """Template for content extraction."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a content extraction expert. Extract and present the most relevant information 
            from web content based on the user's request. Focus on accuracy and relevance."""),
            ("human", """Extract relevant information from this web content:

User request: {request}

Source: {title} ({url})

Content:
{content}

Provide the most relevant information that addresses the user's request.""")
        ])

    def _create_summarize_template(self) -> ChatPromptTemplate:
        """Template for content summarization."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a summarization expert. Create clear, concise summaries of web content 
            that capture the main points and key information."""),
            ("human", """Summarize this web content:

User request: {request}

Source: {title} ({url})

Content:
{content}

Provide a clear summary that addresses what the user is looking for.""")
        ])

    def _create_analyze_template(self) -> ChatPromptTemplate:
        """Template for content analysis."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an analytical expert. Analyze web content deeply, identifying key themes, 
            arguments, evidence, and implications. Provide insights beyond just summarization."""),
            ("human", """Analyze this web content:

User request: {request}

Source: {title} ({url})

Content:
{content}

Provide a thorough analysis that addresses the user's analytical needs.""")
        ])

    def _create_fact_check_template(self) -> ChatPromptTemplate:
        """Template for fact checking."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checking expert. Evaluate the credibility, accuracy, and reliability 
            of information in web content. Identify potential biases, unsupported claims, and factual issues."""),
            ("human", """Fact-check this web content:

User request: {request}

Source: {title} ({url})

Content:
{content}

Evaluate the factual accuracy and credibility of this content.""")
        ])

    def scrape_multiple_urls(self, urls: List[str], analysis_request: str = "") -> List[Dict]:
        """Scrape multiple URLs and return results."""
        results = []
        
        for url in urls:
            try:
                content_data = self._scrape_url(url)
                if content_data:
                    if analysis_request and self.llm:
                        analysis = self._analyze_content(analysis_request, content_data)
                        content_data['analysis'] = analysis
                    results.append(content_data)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                results.append({'url': url, 'error': str(e)})
        
        return results 