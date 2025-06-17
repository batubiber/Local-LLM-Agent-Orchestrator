"""
Main entry point for Local LLM Agent Orchestrator.
"""
import sys
import uvicorn
from src.cli.main import app as cli_app
from src.api.main import app as api_app

def main():
    """
    Main entry point supporting both CLI and API modes.
    Usage:
        - CLI mode: python main.py chat --model-path /path/to/model
        - API mode: python main.py serve --host 0.0.0.0 --port 8000
    """
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # API mode
        host = "0.0.0.0"
        port = 8000
        
        if "--host" in sys.argv:
            idx = sys.argv.index("--host")
            if idx + 1 < len(sys.argv):
                host = sys.argv[idx + 1]
                
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
                
        uvicorn.run(api_app, host=host, port=port)
    else:
        # CLI mode
        cli_app()

if __name__ == "__main__":
    main()
