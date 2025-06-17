"""
Download Mistral 7B GGUF model from Hugging Face.
"""
import os
from pathlib import Path
import requests
from tqdm import tqdm
import typer

app = typer.Typer()

MODELS = {
    "mistral-7b-q4": {
        "name": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "4-bit quantized Mistral 7B (smallest, good balance of size/quality)"
    },
    "mistral-7b-q5": {
        "name": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
        "description": "5-bit quantized Mistral 7B (medium size, better quality)"
    },
    "mistral-7b-q8": {
        "name": "mistral-7b-instruct-v0.2.Q8_0.gguf",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf",
        "description": "8-bit quantized Mistral 7B (largest, best quality)"
    }
}

def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination path
        chunk_size: Download chunk size in bytes
    """
    response = requests.get(url, stream=True, timeout=30)
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            progress.update(size)


@app.command()
def download(
    model: str = typer.Option(
        "mistral-7b-q4",
        help="Model version to download"
    ),
    output_dir: str = typer.Option(
        "models",
        help="Output directory for downloaded model"
    ),
):
    """Download a GGUF model from Hugging Face."""
    if model not in MODELS:
        typer.echo(f"Available models: {', '.join(MODELS.keys())}")
        raise typer.Exit(1)

    model_info = MODELS[model]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dest_path = output_path / model_info["name"]
    if dest_path.exists():
        typer.echo(f"Model already exists at {dest_path}")
        if not typer.confirm("Do you want to download again?"):
            return

    typer.echo(f"\nDownloading {model} model:")
    typer.echo(f"Description: {model_info['description']}")
    typer.echo(f"Destination: {dest_path}\n")

    try:
        download_file(model_info["url"], dest_path)
        typer.echo(f"\nSuccessfully downloaded to {dest_path}")

        # Set the model path environment variable
        os.environ["LLM_MODEL_PATH"] = str(dest_path)
        typer.echo(f"\nSet LLM_MODEL_PATH environment variable to: {dest_path}")

    except Exception as e:
        typer.echo(f"Error downloading model: {e}", err=True)
        if dest_path.exists():
            dest_path.unlink()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
