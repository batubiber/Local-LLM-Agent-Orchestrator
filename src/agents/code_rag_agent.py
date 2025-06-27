"""
Code RAG Agent for analyzing code and providing programming assistance.
"""
from __future__ import annotations

import re
import ast
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CodeRAGAgent(BaseAgent):
    """Agent specialized in code analysis and programming assistance."""

    def __init__(
        self,
        name: str = "code_rag_agent",
        azure_endpoint: str = "",
        azure_api_key: str = "",
        azure_deployment: str = "",
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.1,
        supported_languages: Optional[List[str]] = None
    ) -> None:
        """
        Initialize Code RAG Agent.
        
        Args:
            name: Agent name
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key
            azure_deployment: Azure OpenAI deployment name
            api_version: API version
            temperature: Model temperature (lower for more precise code)
            supported_languages: List of supported programming languages
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
            logger.warning("Azure OpenAI credentials not provided. Code agent will have limited functionality.")

        self.supported_languages = supported_languages or [
            'python', 'javascript', 'typescript', 'java', 'cpp', 'c',
            'csharp', 'go', 'rust', 'ruby', 'php', 'swift', 'kotlin'
        ]

        # Code analysis templates
        self.analysis_templates = {
            "review": self._create_review_template(),
            "explain": self._create_explain_template(),
            "debug": self._create_debug_template(),
            "optimize": self._create_optimize_template(),
            "refactor": self._create_refactor_template(),
            "security": self._create_security_template(),
            "test": self._create_test_template()
        }

    @property
    def name(self) -> str:
        return self._name

    def can_handle(self, request: str) -> bool:
        """Determine if this agent can handle code-related requests."""
        code_keywords = [
            'code', 'function', 'class', 'variable', 'method', 'algorithm',
            'bug', 'debug', 'error', 'exception', 'syntax', 'logic',
            'refactor', 'optimize', 'review', 'test', 'unit test',
            'programming', 'script', 'module', 'library', 'framework',
            'api', 'database', 'sql', 'query', 'regex', 'pattern'
        ]
        
        # Check for programming language mentions
        language_keywords = self.supported_languages + [
            'python', 'javascript', 'java', 'cpp', 'c++', 'html', 'css'
        ]
        
        # Check for code patterns (simplified)
        has_code_pattern = bool(re.search(r'[(){}\[\];]', request)) or '```' in request
        
        request_lower = request.lower()
        has_keywords = any(keyword in request_lower for keyword in code_keywords + language_keywords)
        
        return has_keywords or has_code_pattern

    def process(self, request: str) -> dict:
        """Process the request using the configured strategy."""
        return self.retrieve_and_generate(request)

    def retrieve_and_generate(self, request: str) -> dict:
        """Analyze code and provide programming assistance."""
        if not self.llm:
            return {
                'answer': "Code agent is not properly configured. Please provide Azure OpenAI credentials.",
                'analysis_type': 'error',
                'confidence': 0.0
            }

        # Detect analysis type
        analysis_type = self._detect_analysis_type(request)
        
        # Extract code from request
        code_blocks = self._extract_code_blocks(request)
        
        try:
            # Generate analysis using appropriate template
            template = self.analysis_templates.get(analysis_type, self.analysis_templates['explain'])
            
            # Prepare context
            context = {
                'request': request,
                'code_blocks': '\n'.join(code_blocks) if code_blocks else 'No code provided',
                'language': self._detect_language(request, code_blocks),
                'analysis_type': analysis_type
            }
            
            prompt = template.format_messages(**context)
            response = self.llm.invoke(prompt)
            analysis_result = response.content.strip()

            # Additional analysis if code was provided
            additional_info = {}
            if code_blocks:
                additional_info = self._perform_static_analysis(code_blocks[0]) if code_blocks else {}

            return {
                'answer': analysis_result,
                'analysis_type': analysis_type,
                'language': context['language'],
                'confidence': 0.85,
                'code_blocks_found': len(code_blocks),
                **additional_info
            }

        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {
                'answer': f"Error analyzing code: {str(e)}",
                'analysis_type': analysis_type,
                'confidence': 0.0
            }

    def summarize(self, request: str) -> dict:
        """Generate a code summary for the given request."""
        return self.retrieve_and_generate(request)

    def _detect_analysis_type(self, request: str) -> str:
        """Detect the type of code analysis requested."""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['review', 'check', 'audit', 'quality']):
            return 'review'
        elif any(word in request_lower for word in ['debug', 'fix', 'error', 'bug', 'issue', 'problem']):
            return 'debug'
        elif any(word in request_lower for word in ['optimize', 'performance', 'speed', 'efficiency']):
            return 'optimize'
        elif any(word in request_lower for word in ['refactor', 'clean', 'improve', 'restructure']):
            return 'refactor'
        elif any(word in request_lower for word in ['security', 'secure', 'vulnerability', 'safe']):
            return 'security'
        elif any(word in request_lower for word in ['test', 'testing', 'unit test', 'pytest']):
            return 'test'
        else:
            return 'explain'

    def _extract_code_blocks(self, request: str) -> List[str]:
        """Extract code blocks from the request."""
        code_blocks = []
        
        # Look for markdown code blocks
        markdown_pattern = r'```(?:[\w]*\n)?(.*?)```'
        markdown_matches = re.findall(markdown_pattern, request, re.DOTALL)
        code_blocks.extend([match.strip() for match in markdown_matches])
        
        # Look for inline code
        inline_pattern = r'`([^`]+)`'
        inline_matches = re.findall(inline_pattern, request)
        code_blocks.extend(inline_matches)
        
        return code_blocks

    def _detect_language(self, request: str, code_blocks: List[str]) -> str:
        """Detect the programming language."""
        request_lower = request.lower()
        
        # Check explicit language mentions
        for lang in self.supported_languages:
            if lang in request_lower:
                return lang
        
        # Analyze code blocks for language indicators
        if code_blocks:
            code = code_blocks[0]
            
            # Python indicators
            if any(keyword in code for keyword in ['def ', 'import ', 'from ', 'print(', '__init__']):
                return 'python'
            
            # JavaScript/TypeScript indicators
            if any(keyword in code for keyword in ['function ', 'const ', 'let ', 'var ', '=>', 'console.log']):
                return 'javascript'
            
            # Java indicators
            if any(keyword in code for keyword in ['public class', 'private ', 'System.out']):
                return 'java'
            
            # C/C++ indicators
            if any(keyword in code for keyword in ['#include', 'int main', 'printf', 'cout']):
                return 'cpp'
        
        return 'unknown'

    def _perform_static_analysis(self, code: str) -> Dict[str, Any]:
        """Perform basic static analysis on the code."""
        analysis = {
            'lines_of_code': len(code.split('\n')),
            'character_count': len(code),
            'complexity_indicators': []
        }
        
        # Basic complexity indicators
        complexity_patterns = {
            'nested_loops': r'for.*for|while.*while|for.*while|while.*for',
            'many_conditions': r'if.*and.*or|if.*or.*and',
        }
        
        for indicator, pattern in complexity_patterns.items():
            if pattern and re.search(pattern, code, re.IGNORECASE):
                analysis['complexity_indicators'].append(indicator)
        
        return analysis

    def _create_review_template(self) -> ChatPromptTemplate:
        """Template for code review."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert code reviewer. Analyze the provided code for:
            - Code quality and best practices
            - Potential bugs and issues
            - Performance considerations
            - Maintainability and readability
            - Design patterns and architecture
            Provide constructive feedback with specific suggestions for improvement."""),
            ("human", """Please review this code:

Language: {language}
Analysis Type: {analysis_type}

Request: {request}

Code to review:
{code_blocks}

Provide a thorough code review with specific recommendations.""")
        ])

    def _create_explain_template(self) -> ChatPromptTemplate:
        """Template for code explanation."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a programming instructor who explains code clearly and thoroughly. 
            Break down the code step by step, explaining:
            - What each part does
            - How it works
            - Why it's structured that way
            - Any important concepts or patterns used
            Make your explanation accessible while being technically accurate."""),
            ("human", """Please explain this code:

Language: {language}
Request: {request}

Code to explain:
{code_blocks}

Provide a clear, educational explanation of how this code works.""")
        ])

    def _create_debug_template(self) -> ChatPromptTemplate:
        """Template for debugging assistance."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert debugger. Help identify and fix issues in code by:
            - Analyzing potential bugs and errors
            - Suggesting specific fixes
            - Explaining why issues occur
            - Providing prevention strategies
            - Offering alternative approaches if needed
            Be systematic and thorough in your debugging approach."""),
            ("human", """Please help debug this code:

Language: {language}
Request: {request}

Code with issues:
{code_blocks}

Identify problems and provide specific solutions.""")
        ])

    def _create_optimize_template(self) -> ChatPromptTemplate:
        """Template for code optimization."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a performance optimization expert. Analyze code for:
            - Performance bottlenecks
            - Efficiency improvements
            - Memory usage optimization
            - Algorithm improvements
            - Best practices for speed
            Provide specific optimization suggestions with explanations."""),
            ("human", """Please optimize this code:

Language: {language}
Request: {request}

Code to optimize:
{code_blocks}

Suggest performance improvements and optimizations.""")
        ])

    def _create_refactor_template(self) -> ChatPromptTemplate:
        """Template for code refactoring."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a refactoring expert. Help improve code structure by:
            - Improving readability and maintainability
            - Applying design patterns appropriately
            - Reducing complexity and duplication
            - Enhancing modularity
            - Following coding standards
            Provide refactored code with explanations."""),
            ("human", """Please refactor this code:

Language: {language}
Request: {request}

Code to refactor:
{code_blocks}

Provide improved, refactored code with explanations.""")
        ])

    def _create_security_template(self) -> ChatPromptTemplate:
        """Template for security analysis."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a security expert. Analyze code for:
            - Security vulnerabilities
            - Input validation issues
            - Authentication and authorization flaws
            - Data exposure risks
            - Injection attacks
            Provide specific security recommendations and fixes."""),
            ("human", """Please analyze this code for security issues:

Language: {language}
Request: {request}

Code to analyze:
{code_blocks}

Identify security vulnerabilities and provide fixes.""")
        ])

    def _create_test_template(self) -> ChatPromptTemplate:
        """Template for test generation."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a test automation expert. Help create comprehensive tests by:
            - Writing unit tests for the given code
            - Covering edge cases and error conditions
            - Following testing best practices
            - Using appropriate testing frameworks
            - Ensuring good test coverage
            Provide complete, runnable test code."""),
            ("human", """Please create tests for this code:

Language: {language}
Request: {request}

Code to test:
{code_blocks}

Generate comprehensive unit tests.""")
        ])

    def analyze_codebase(self, directory_path: str) -> dict:
        """Analyze an entire codebase directory."""
        try:
            path = Path(directory_path)
            if not path.exists():
                return {'error': f"Directory {directory_path} does not exist"}
            
            code_files = []
            for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs']:
                code_files.extend(path.rglob(f'*{ext}'))
            
            analysis = {
                'total_files': len(code_files),
                'languages': {},
                'total_lines': 0,
                'file_analysis': []
            }
            
            for file_path in code_files[:20]:  # Limit to first 20 files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_info = {
                        'file': str(file_path.relative_to(path)),
                        'lines': len(content.split('\n')),
                        'size': len(content)
                    }
                    
                    # Detect language
                    lang = self._detect_language("", [content])
                    if lang != 'unknown':
                        analysis['languages'][lang] = analysis['languages'].get(lang, 0) + 1
                    
                    analysis['total_lines'] += file_info['lines']
                    analysis['file_analysis'].append(file_info)
                    
                except Exception as e:
                    logger.warning(f"Could not analyze file {file_path}: {e}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing codebase: {e}")
            return {'error': str(e)} 