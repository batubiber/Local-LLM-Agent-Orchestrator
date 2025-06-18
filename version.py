"""
Versiyon bilgisi
"""
from dataclasses import dataclass

@dataclass
class VersiyonBilgisi:
    """
    Versiyon bilgisi
    """
    major_versiyon  : int = 0
    minor_versiyon  : int = 2
    build_versiyon  : int = 0
    product_id      : int = 0

    def __str__(self):
        return f"{self.major_versiyon}.{self.minor_versiyon}.{self.build_versiyon}.{self.product_id}"

    def __repr__(self):
        return self.__str__()

# VERSIYON GECMISI

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
