"""
Command-line interface for Local LLM Agent Orchestrator.
"""
from __future__ import annotations

import os
import sys
import signal
import numpy as np
from typing import Optional, List
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.app import create_orchestrator
from src.data.vector_store import VectorStore
from src.data.document_processor import DocumentProcessor
from src.models import RAGConfig

app = typer.Typer(
    name="llm-agent",
    help="Local LLM Agent Orchestrator CLI",
    add_completion=False,
)
console = Console()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nExiting gracefully...")
    sys.exit(0)


@app.command()
def chat(
    context_dir: str = typer.Option(
        "context",
        "--context-dir",
        "-c",
        help="Directory containing context documents",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Model temperature",
    ),
):
    """Start an interactive chat session."""
    try:
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, signal_handler)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading model and initializing agents...", total=None)
            orchestrator = create_orchestrator(
                temperature=temperature,
                vector_db_path=context_dir
            )

        console.print("\n[green]Model loaded successfully! Start chatting (Ctrl+C to exit)[/green]\n")
        
        print("\nWelcome to the Local LLM Agent Orchestrator!")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'help' for available commands.\n")

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit']:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break
                
                # Check for help command
                if user_input.lower() == 'help':
                    console.print("\n[blue]Available commands:[/blue]")
                    console.print("- 'exit' or 'quit': End the session")
                    console.print("- 'help': Show this help message")
                    console.print("- Any other input will be processed as a query")
                    continue

                # Process the input
                if user_input:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        progress.add_task("Thinking...", total=None)
                        response = orchestrator.handle(user_input)

                    console.print("\n[blue]Assistant:[/blue]")
                    # Handle different types of responses
                    if "rag" in response:
                        response_data = response["rag"]
                        if "answer" in response_data:
                            console.print(response_data["answer"])
                        if "contexts" in response_data:
                            console.print("\n[dim]Sources:[/dim]")
                            for ctx in response_data["contexts"]:
                                console.print(f"- {ctx.get('source', 'Unknown')} (Page {ctx.get('page', 'N/A')})")
                    elif "summary" in response:
                        response_data = response["summary"]
                        if "summary" in response_data:
                            console.print(response_data["summary"])
                    else:
                        console.print("I'm not sure how to handle that request. Try asking for a search or summary.")
                    console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting gracefully...[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]\n")

    except Exception as e:
        console.print(f"\n[red]Failed to initialize: {str(e)}[/red]")
        sys.exit(1)


@app.command()
def ingest(
    input_dir: str = typer.Argument(
        ...,
        help="Directory containing documents to process",
    ),
    output_dir: str = typer.Option(
        "context",
        "--output-dir",
        "-o",
        help="Output directory for processed files",
    ),
):
    """Process and index documents for RAG."""
    try:
        # Create context directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize RAG config
        rag_config = RAGConfig()

        # Initialize document processor
        processor = DocumentProcessor()
        
        # Process the document
        print(f"Processing {input_dir}...")
        chunks = list(processor.process_pdf(input_dir))
        
        # Initialize vector store with proper dimension
        vector_store = VectorStore(
            dimension=rag_config.vector_dimension,
            index_path=os.path.join(output_dir, "faiss.index"),
            embedding_model=rag_config.embedding_model
        )
        
        # Add chunks to vector store
        for chunk in chunks:
            chunk = processor.compute_embedding(chunk)
            # Convert embedding to numpy array and reshape
            embedding = np.array(chunk.embedding, dtype=np.float32).reshape(1, -1)
            vector_store.add_vectors(
                vectors=embedding,
                metadata=[chunk.metadata]
            )
        
        # Save the vector store
        vector_store.save()
        
        print(f"Successfully processed {len(chunks)} chunks from {input_dir}")

    except Exception as e:
        console.print(f"\n[red]Failed to process documents: {str(e)}[/red]")
        sys.exit(1)


def main():
    """CLI entry point."""
    app()
