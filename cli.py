#!/usr/bin/env python
"""
Command-line interface for Depression Detection system.
Allows analyzing audio files for depression biomarkers.
"""

import argparse
import asyncio
import json
import os
import uuid
from typing import Dict, Any
import wave
import logging
import sys

from depression_detector import DepressionDetector
from data_storage import DepressionDataStorage
from config import settings
from logging_config import setup_logging, get_default_log_file

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource
)

# Set up logging
logger = setup_logging(log_file=get_default_log_file())

def analyze_audio_file(file_path: str, api_key: str = None) -> Dict[str, Any]:
    """
    Analyze an audio file for depression biomarkers.
    
    Args:
        file_path: Path to audio file
        api_key: Deepgram API key (optional)
        
    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    
    # Get API key from settings or parameter
    deepgram_api_key = api_key or settings.DEEPGRAM_API_KEY
    if not deepgram_api_key:
        return {"error": "Deepgram API key not found. Set in .env file or pass as parameter."}
    
    # Initialize depression detector
    detector = DepressionDetector()
    
    # Initialize data storage
    storage = DepressionDataStorage()
    session_id = str(uuid.uuid4())
    storage.create_session(session_id)
    
    try:
        # Process file with Deepgram
        async def process_audio():
            # Initialize Deepgram client
            deepgram = DeepgramClient(deepgram_api_key)
            
            # Set up options
            options = PrerecordedOptions(
                model=settings.DEEPGRAM_MODEL,
                language=settings.DEEPGRAM_LANGUAGE,
                smart_format=True
            )
            
            # Transcribe audio file
            with open(file_path, "rb") as audio:
                source = FileSource(audio)
                response = await deepgram.listen.prerecorded.v("1").transcribe_file(source, options)
                transcript_response = response.to_dict()
            
            # Extract transcript
            if "results" in transcript_response and "channels" in transcript_response["results"]:
                channel = transcript_response["results"]["channels"][0]
                if "alternatives" in channel and len(channel["alternatives"]) > 0:
                    transcript = channel["alternatives"][0].get("transcript", "")
                    
                    if transcript.strip():
                        # Analyze for depression
                        depression_score, features = detector.analyze_text(transcript)
                        depression_level = detector.get_depression_level(depression_score)
                        feedback = detector.get_feedback(depression_score, features)
                        
                        # Save result to database
                        storage.save_analysis_result(
                            session_id, 
                            transcript, 
                            depression_score, 
                            depression_level,
                            features
                        )
                        
                        # Update session info
                        storage.update_session(session_id, 0, 1)
                        
                        return {
                            "transcript": transcript,
                            "depression_score": depression_score,
                            "depression_level": depression_level,
                            "feedback": feedback,
                            "features": features,
                            "session_id": session_id
                        }
            
            return {"error": "Failed to extract transcript from audio"}
        
        # Run the async function
        return asyncio.run(process_audio())
    
    except Exception as e:
        logger.error(f"Error analyzing audio file: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Depression Detection System - CLI")
    parser.add_argument("--file", "-f", help="Audio file to analyze", required=True)
    parser.add_argument("--api-key", "-k", help="Deepgram API key (required unless set in .env file)")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Add API key explanation in epilog
    parser.epilog = """
    API Key Information:
    -------------------
    A Deepgram API key is required to use this application. You can provide it in two ways:
    1. Pass it using the --api-key parameter
    2. Set it in your .env file as DEEPGRAM_API_KEY=your_key_here
    
    Get a free API key at: https://console.deepgram.com/signup
    """
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Analyze audio file
    print(f"Analyzing audio file: {args.file}")
    results = analyze_audio_file(args.file, args.api_key)
    
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
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Session info
    print(f"\nSession ID: {results['session_id']}")
    print("Use this session ID to retrieve results from the database later.")

if __name__ == "__main__":
    main()
