#!/usr/bin/env python3
"""
Main entry point for the CrewAI agent with A2A protocol
"""

import argparse
import os
import uvicorn
from dotenv import load_dotenv

from agent import setup_a2a_server

# Load environment variables from .env file
load_dotenv()

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Run the CrewAI image generation agent with A2A protocol")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=10001, help="Port to bind the server to")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory to store cached images")
    args = parser.parse_args()

    # Check if OpenAI API key is present
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it in a .env file or export it in your shell")
        return 1

    # Start the A2A server
    app = setup_a2a_server(cache_dir=args.cache_dir)
    
    print(f"Starting A2A server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
