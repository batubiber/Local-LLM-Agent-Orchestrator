"""
Version information
"""
from dataclasses import dataclass

@dataclass
class VersionInfo:
    """
    Version information
    """
    major_version  : int = 0
    minor_version  : int = 3
    build_version  : int = 0
    product_id     : int = 0

    def __str__(self):
        return f"{self.major_version}.{self.minor_version}.{self.build_version}.{self.product_id}"

    def __repr__(self):
        return self.__str__()

# VERSION HISTORY

####################################################################
# Version    : 3.0.0
# Developers : Batuhan Biber
# Date       : 27.06.2025
#
# Developments:
# - SummaryAgent - Multi-Type Summary Generation
#   * Executive, technical, narrative, bullet-point, and abstract summaries
#   * Intelligent summary type detection
#   * Full API integration
# - WebScrapingAgent - Web Scraping and Analysis
#   * Web page content extraction and analysis
#   * Multi-URL processing
#   * Fact-checking and summarization
# - CodeRAGAgent - Code Analysis and Review
#   * Code review, debugging, optimization, refactoring, security analysis
#   * Multi-language support (Python, JavaScript, Java, C++, etc.)
#   * Test generation capabilities
# - REST API Layer - Complete API Access
#   * FastAPI-based REST API with OpenAPI documentation
#   * Health monitoring and status endpoints
#   * CORS support for web integration
# - Agent Template System - Standardized Creation
#   * Complete template for rapid agent development
#   * Best practices and examples included
#   * Integration guidelines
# - Performance Monitoring - Real-time Tracking
#   * Success/failure rates, response times, confidence scores
#   * Agent health monitoring
#   * Metrics export capabilities
#
####################################################################

####################################################################
# Version    : 2.0.0
# Developers : Batuhan Biber
# Date       : 18.06.2025
#
# Developments:
# - Migrated to Azure OpenAI from local LLM models
# - Removed local model infrastructure (models/ and scripts/)
# - Updated dependencies to latest versions in pyproject.toml and
#   requirements.txt
# - Improved project structure and documentation
# - Added uv package manager support with detailed instructions
# - Cleaned up unnecessary files and directories
# - Updated .gitignore for better project organization
# - Enhanced README with clearer installation and usage instructions
#
####################################################################

####################################################################
# Version    : 1.1.0
# Developers : Batuhan Biber
# Date       : 17.06.2025
#
# Developments:
# - Fixed vector store initialization in ingest command by properly
#   configuring dimension and index path
# - Added proper numpy array conversion for embeddings before adding
#   to vector store
# - Ensured embeddings are reshaped to correct format (1 x dimension)
# - Added vector store saving after processing documents
# - Improved error handling and logging in document processing pipeline
#
####################################################################

####################################################################
# Version    : 1.0.0
# Developers : Batuhan Biber
# Date       : 17.06.2025
#
# Developments:
# - initial commit
#
####################################################################
