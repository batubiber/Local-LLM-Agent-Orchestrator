"""
GraphRAG CLI interface with Milvus Vector DB and Azure OpenAI.
"""
from __future__ import annotations

import os
import sys
import signal
import logging
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from dotenv import load_dotenv

from src.orchestrator.graph_app import create_graph_rag_orchestrator_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    print("Warning: .env file not found at", env_path)

app = typer.Typer(
    name="graph-rag",
    help="GraphRAG CLI with Milvus Vector DB and Azure OpenAI",
    add_completion=False,
)
console = Console()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    console.print("\n[yellow]Exiting gracefully...[/yellow]")
    sys.exit(0)


@app.command()
def chat(
    context_dir: str = typer.Option(
        "context",
        "--context-dir",
        "-c",
        help="Directory containing context documents",
    ),
    embedding_model: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--embedding-model",
        "-e",
        help="Sentence transformer embedding model",
    ),
    temperature: float = typer.Option(
        0.0,
        "--temperature",
        "-t",
        help="Model temperature",
    ),
    use_llm_extraction: bool = typer.Option(
        True,
        "--use-llm/--use-rules",
        help="Use LLM for triplet extraction (vs rule-based)",
    ),
):
    """Start an interactive GraphRAG chat session."""
    try:
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)

        console.print("\n[bold blue]GraphRAG System with Milvus Vector DB[/bold blue]")
        console.print("Initializing system components...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading GraphRAG components...", total=None)
            
            orchestrator = create_graph_rag_orchestrator_from_env(
                context_dir=context_dir,
                embedding_model=embedding_model,
                temperature=temperature,
                use_llm_extraction=use_llm_extraction
            )

        console.print("\n[green]GraphRAG system loaded successfully![/green]")
        console.print("[dim]Type 'exit', 'quit', or press Ctrl+C to end the session.[/dim]")
        console.print("[dim]Type 'help' for available commands.[/dim]")
        console.print("[dim]Type 'compare <query>' to compare GraphRAG vs Naive RAG.[/dim]\n")

        while True:
            try:
                # Get user input
                user_input = typer.prompt("\n🤖 You").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit']:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break
                
                # Check for help command
                if user_input.lower() == 'help':
                    console.print("\n[blue]Available commands:[/blue]")
                    console.print("• 'exit' or 'quit': End the session")
                    console.print("• 'help': Show this help message")
                    console.print("• 'compare <query>': Compare GraphRAG with Naive RAG")
                    console.print("• Any other input will be processed as a GraphRAG query")
                    continue

                # Check for compare command
                if user_input.lower().startswith('compare '):
                    query = user_input[8:].strip()  # Remove 'compare ' prefix
                    if query:
                        console.print(f"\n[blue]Comparing responses for:[/blue] {query}")
                        
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=console,
                        ) as progress:
                            progress.add_task("Running comparison...", total=None)
                            
                            # Get the GraphRAG agent for comparison
                            graph_rag_agent = orchestrator._agents[0]  # First (and only) agent
                            comparison_result = graph_rag_agent.compare_with_naive_rag(query)
                        
                        if 'error' in comparison_result:
                            console.print(f"\n[red]Error: {comparison_result['error']}[/red]")
                        else:
                            console.print("\n[bold green]GraphRAG Result:[/bold green]")
                            console.print(comparison_result['graph_rag']['answer'])
                            
                            console.print("\n[bold red]Naive RAG Result:[/bold red]")
                            console.print(comparison_result['naive_rag']['answer'])
                            
                            console.print("\n[dim]GraphRAG passages:[/dim]")
                            for i, passage in enumerate(comparison_result['graph_rag']['passages'], 1):
                                console.print(f"[dim]{i}. {passage[:200]}...[/dim]")
                            
                            console.print("\n[dim]Naive RAG passages:[/dim]")
                            for i, passage in enumerate(comparison_result['naive_rag']['passages'], 1):
                                console.print(f"[dim]{i}. {passage[:200]}...[/dim]")
                    else:
                        console.print("[red]Please provide a query after 'compare'[/red]")
                    continue

                # Process regular query
                if user_input:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        progress.add_task("Processing with GraphRAG...", total=None)
                        response = orchestrator.handle(user_input)

                    console.print("\n[blue]🤖 GraphRAG Assistant:[/blue]")
                    
                    # Handle GraphRAG response
                    if "graph_rag" in response:
                        response_data = response["graph_rag"]
                        if "answer" in response_data:
                            console.print(response_data["answer"])
                            
                        if "contexts" in response_data and response_data["contexts"]:
                            console.print("\n[dim]📚 Sources:[/dim]")
                            for i, ctx in enumerate(response_data["contexts"], 1):
                                source = ctx.get('source', f'Document {i}')
                                score = ctx.get('score', 0.0)
                                console.print(f"[dim]{i}. {source} (score: {score:.2f})[/dim]")
                    else:
                        console.print("I'm not sure how to handle that request. Try asking a different question.")

            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting gracefully...[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                continue

    except Exception as e:
        console.print(f"\n[red]Failed to initialize GraphRAG system: {str(e)}[/red]")
        console.print("\n[yellow]Please ensure the following environment variables are set:[/yellow]")
        console.print("• MILVUS_URI: Your Milvus connection URI")
        console.print("• MILVUS_TOKEN: Your Milvus authentication token (optional for local)")
        console.print("• AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint")
        console.print("• AZURE_OPENAI_API_KEY: Your Azure OpenAI API key")
        console.print("• AZURE_OPENAI_DEPLOYMENT_NAME: Your Azure OpenAI deployment name")
        sys.exit(1)


@app.command()
def info():
    """Show information about the GraphRAG system."""
    console.print("\n[bold blue]GraphRAG System Information[/bold blue]")
    console.print("==================================")
    console.print("🔧 Components:")
    console.print("  • Vector Database: Milvus")
    console.print("  • LLM Provider: Azure OpenAI")
    console.print("  • Embedding Model: Sentence Transformers")
    console.print("  • Graph Reasoning: Multi-hop expansion with adjacency matrices")
    console.print("  • Reranking: LLM-based with Chain-of-Thought prompting")
    
    console.print("\n📋 Required Environment Variables:")
    console.print("  • MILVUS_URI")
    console.print("  • MILVUS_TOKEN (optional for local)")
    console.print("  • AZURE_OPENAI_ENDPOINT")
    console.print("  • AZURE_OPENAI_API_KEY")
    console.print("  • AZURE_OPENAI_DEPLOYMENT_NAME")
    
    console.print("\n🚀 Features:")
    console.print("  • Multi-hop reasoning across knowledge graphs")
    console.print("  • Entity and relationship extraction")
    console.print("  • Semantic similarity search with graph expansion")
    console.print("  • LLM-powered intelligent reranking")
    console.print("  • Comparison with traditional RAG methods")
    
    console.print("\n📖 Usage:")
    console.print("  • Place PDF documents in the context directory")
    console.print("  • Run 'graph-rag chat' to start interactive session")
    console.print("  • Use 'compare <query>' to compare with naive RAG")


@app.command()
def test():
    """Test the GraphRAG system with sample queries."""
    try:
        console.print("\n[bold blue]Testing GraphRAG System[/bold blue]")
        console.print("========================")
        
        # Test queries similar to the notebook examples
        test_queries = [
            "What contribution did the son of Euler's teacher make?",
            "Who was Johann Bernoulli's teacher?",
            "What is the relationship between Daniel Bernoulli and fluid dynamics?",
            "How are the Bernoulli family and Euler connected?",
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing GraphRAG system...", total=None)
            
            orchestrator = create_graph_rag_orchestrator_from_env(
                context_dir="context",
                use_llm_extraction=False  # Use rule-based for faster testing
            )
        
        console.print("\n[green]Running test queries...[/green]\n")
        
        for i, query in enumerate(test_queries, 1):
            console.print(f"[bold]Test {i}: {query}[/bold]")
            
            try:
                response = orchestrator.handle(query)
                
                if "graph_rag" in response:
                    answer = response["graph_rag"].get("answer", "No answer generated")
                    console.print(f"[green]Answer:[/green] {answer}")
                else:
                    console.print("[red]No response generated[/red]")
                    
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
            
            console.print("-" * 50)
            
        console.print("\n[green]Test completed![/green]")
        
    except Exception as e:
        console.print(f"\n[red]Test failed: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app() 