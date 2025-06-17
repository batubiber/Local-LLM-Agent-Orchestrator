"""
Command-line interface for Local LLM Agent Orchestrator.
"""
from __future__ import annotations

import os
from typing import Optional, List
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.app import create_orchestrator

app = typer.Typer(
    name="llm-agent",
    help="Local LLM Agent Orchestrator CLI",
    add_completion=False,
)
console = Console()


@app.command()
def chat(
    model_path: str = typer.Option(
        ...,
        "--model-path",
        "-m",
        help="Path to local LLM model",
    ),
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
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading model and initializing agents...", total=None)
            orchestrator = create_orchestrator(
                model_path=model_path,
                temperature=temperature,
                vector_db_path=context_dir
            )

        console.print("\n[green]Model loaded successfully! Start chatting (Ctrl+C to exit)[/green]\n")
        
        while True:
            try:
                query = typer.prompt("You")
                if not query.strip():
                    continue

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task("Thinking...", total=None)
                    response = orchestrator.handle(query)

                console.print("\n[blue]Assistant:[/blue]")
                console.print(response["response"])
                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]\n")

    except Exception as e:
        console.print(f"\n[red]Failed to initialize: {str(e)}[/red]")
        raise typer.Exit(1)


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
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents...", total=None)

            # Document processing logic here
            # TODO: Implement document processing

            progress.update(task, description="Documents processed successfully!")

    except Exception as e:
        console.print(f"\n[red]Failed to process documents: {str(e)}[/red]")
        raise typer.Exit(1)


def main():
    """CLI entry point."""
    app()
