#!/usr/bin/env python
"""
Entry point script for Depression Detection System.
Provides a simple way to start the web server or use the CLI.
"""

import argparse
import os
import sys
import logging
import uvicorn
from logging_config import setup_logging, get_default_log_file
from config import settings

# Set up logging
logger = setup_logging(log_file=get_default_log_file())

def main():
    """Main entry point function."""
    parser = argparse.ArgumentParser(description="Depression Detection System")
    
    # Define command groups
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Web server command
    server_parser = subparsers.add_parser("server", help="Run the web server")
    server_parser.add_argument("--host", type=str, default=settings.HOST, help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=settings.PORT, help="Port to bind to")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    # CLI command
    cli_parser = subparsers.add_parser("analyze", help="Analyze an audio file")
    cli_parser.add_argument("--file", "-f", required=True, help="Audio file to analyze")
    cli_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    cli_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Handle commands
    if args.command == "server":
        # Run web server
        logger.info(f"Starting web server on {args.host}:{args.port}")
        print(f"Starting depression detection web server at http://{args.host}:{args.port}")
        uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)
    
    elif args.command == "analyze":
        # Import CLI module and run analysis
        from cli import analyze_audio_file
        
        # Set log level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Analyze audio file
        print(f"Analyzing audio file: {args.file}")
        results = analyze_audio_file(args.file)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            sys.exit(1)
        
        # Display results
        print("\n=== Depression Analysis Results ===")
        print(f"Transcript: {results['transcript'][:100]}...")
        print(f"Depression Score: {results['depression_score']:.1f}/100")
        print(f"Depression Level: {results['depression_level'].upper()}")
        print("\nDetailed Feedback:")
        print(results['feedback'])
        
        # Save results to file if requested
        if args.output:
            import json
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
