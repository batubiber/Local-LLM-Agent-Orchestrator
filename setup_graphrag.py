#!/usr/bin/env python3
"""
Setup script for GraphRAG system with Milvus Vector DB and Azure OpenAI.
"""
import sys
from pathlib import Path
import subprocess
import spacy

def create_env_file():
    """Create .env file with configuration template."""
    env_content = """# GraphRAG System Configuration

# ===== Milvus Vector Database =====
# For Zilliz Cloud (recommended)
MILVUS_URI=your_milvus_uri
MILVUS_TOKEN=your_milvus_token

# For local Milvus (alternative)
# MILVUS_URI=http://localhost:19530
# MILVUS_TOKEN=  # Leave empty for local Milvus

# ===== Azure OpenAI Configuration =====
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# ===== Optional Settings =====
# Embedding model for document processing
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Model temperature for generation
TEMPERATURE=0.0

# Use LLM for triplet extraction (true) or rule-based extraction (false)
USE_LLM_EXTRACTION=true
"""

    env_path = Path(".env")
    if env_path.exists():
        print("⚠️  .env file already exists. Skipping creation.")
        return False

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_content)

    print("✅ Created .env file with configuration template")
    return True


def create_context_directory():
    """Create context directory for documents."""
    context_dir = Path("context")
    context_dir.mkdir(exist_ok=True)

    # Create a sample file with instructions
    sample_file = context_dir / "README.txt"
    if not sample_file.exists():
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("""GraphRAG Context Directory
========================

Place your PDF documents in this directory for processing.

The GraphRAG system will:
1. Extract text from PDF files
2. Generate knowledge graph triplets using Azure OpenAI
3. Create entity and relationship embeddings
4. Enable multi-hop reasoning queries

Supported formats:
- PDF files (.pdf)

Example usage:
1. Copy your PDF documents to this directory
2. Run: python -m src.cli.graph_rag_cli chat
3. Ask complex questions that require multi-hop reasoning

For best results:
- Use documents with clear entity relationships
- Academic papers, biographical texts, and technical documentation work well
- The system excels at questions like "What contribution did X's student make?"
""")

    print(f"✅ Created context directory: {context_dir}")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        # Verify spaCy model
        try:
            _ = spacy.load("en_core_web_sm")
            print("✅ spaCy model verified (already installed)")
        except OSError:
            print("📥 spaCy model 'en_core_web_sm' not found. Installing...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            print("✅ spaCy model downloaded and installed successfully")

        print("✅ Core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: uv pip install -r requirements.txt to install all dependencies")
        return False


def print_setup_instructions():
    """Print setup instructions for the user."""
    print("\n" + "="*60)
    print("🚀 GraphRAG System Setup Complete!")
    print("="*60)

    print("\n📋 Next Steps:")
    print("1. Edit the .env file with your credentials:")
    print("   - Get Milvus URI and token from Zilliz Cloud")
    print("   - Add your Azure OpenAI endpoint and API key")

    print("\n2. Add PDF documents to the context/ directory")

    print("\n3. Start the GraphRAG system:")
    print("   python -m src.cli.graph_rag_cli chat")

    print("\n4. Try example queries:")
    print("   - What contribution did the son of Euler's teacher make?")
    print("   - How are Daniel Bernoulli and fluid dynamics related?")

    print("\n📚 Additional Commands:")
    print("   python -m src.cli.graph_rag_cli info     # System information")
    print("   python -m src.cli.graph_rag_cli test     # Run test queries")

    print("\n🔧 Configuration:")
    print("   - Azure OpenAI is used for triplet extraction and reasoning")
    print("   - Use 'compare <query>' in chat to compare with naive RAG")
    print("   - Adjust temperature and other settings in .env file")

    print("\n📖 Documentation:")
    print("   - See README.md for detailed instructions")
    print("   - Check .env file for configuration options")
    print("   - Visit Azure OpenAI documentation for deployment setup")


def main():
    """Main setup function."""
    print("🔧 Setting up GraphRAG System with Azure OpenAI")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        print("\n💡 Install dependencies using: uv pip install -r requirements.txt")
        return

    # Create configuration files and directories
    env_created = create_env_file()
    create_context_directory()

    # Print instructions
    print_setup_instructions()

    if env_created:
        print("\n⚠️  IMPORTANT: Edit the .env file with your Azure OpenAI and Milvus credentials before running!")


if __name__ == "__main__":
    main()
