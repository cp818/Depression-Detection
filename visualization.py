"""
Visualization utilities for the Depression Detection system.
Provides functions for generating charts and visualizations of depression analysis results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from data_storage import DepressionDataStorage

logger = logging.getLogger(__name__)

def create_depression_score_chart(scores: List[float], timestamps: Optional[List[str]] = None, 
                                 title: str = "Depression Score Trend", 
                                 output_path: Optional[str] = None) -> str:
    """
    Create a chart showing depression score trend over time.
    
    Args:
        scores: List of depression scores
        timestamps: Optional list of timestamps
        title: Chart title
        output_path: Optional output file path
        
    Returns:
        Path to saved chart image
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Create x-axis (either timestamps or sequential numbers)
        x = range(len(scores))
        if timestamps:
            # Convert to datetime objects for better formatting
            if len(timestamps) == len(scores):
                try:
                    x = [datetime.fromisoformat(ts) for ts in timestamps]
                except ValueError:
                    # If conversion fails, use sequential numbers
                    x = range(len(scores))
        
        # Plot the scores
        plt.plot(x, scores, marker='o', linestyle='-', color='#2c7fb8', linewidth=2, markersize=6)
        
        # Add colored background regions for different depression levels
        plt.axhspan(0, 20, alpha=0.2, color='green', label='Low Risk')
        plt.axhspan(20, 40, alpha=0.2, color='yellow', label='Mild Risk')
        plt.axhspan(40, 60, alpha=0.2, color='orange', label='Moderate Risk')
        plt.axhspan(60, 80, alpha=0.2, color='red', label='High Risk')
        plt.axhspan(80, 100, alpha=0.2, color='purple', label='Severe Risk')
        
        # Configure the chart
        plt.ylim(0, 100)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Depression Score', fontsize=12)
        
        if isinstance(x[0], datetime):
            plt.xlabel('Time', fontsize=12)
            plt.gcf().autofmt_xdate()
        else:
            plt.xlabel('Sample', fontsize=12)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        
        # Save the chart if output path is provided
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            return output_path
        else:
            # Create a temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = "temp_charts"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = f"{temp_dir}/depression_score_chart_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(temp_path, dpi=300)
            plt.close()
            return temp_path
    
    except Exception as e:
        logger.error(f"Failed to create depression score chart: {e}")
        return ""

def create_feature_radar_chart(features: Dict[str, float], 
                              title: str = "Depression Features", 
                              output_path: Optional[str] = None) -> str:
    """
    Create a radar chart visualizing different depression features.
    
    Args:
        features: Dictionary of feature names and values (0-1 scale)
        title: Chart title
        output_path: Optional output file path
        
    Returns:
        Path to saved chart image
    """
    try:
        # Ensure we have features
        if not features:
            logger.error("No features provided for radar chart")
            return ""
        
        # Create figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Extract labels and values, ensuring values are in 0-1 range
        labels = list(features.keys())
        values = [min(max(v, 0), 1) for v in features.values()]
        
        # Number of variables
        N = len(labels)
        
        # Compute angle for each feature
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        
        # Make the plot circular by repeating the first value
        values += [values[0]]
        angles += [angles[0]]
        labels += [labels[0]]
        
        # Plot features
        ax.plot(angles, values, 'o-', linewidth=2, color='#d73027')
        ax.fill(angles, values, alpha=0.25, color='#d73027')
        
        # Set labels and features
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        
        # Configure chart
        ax.set_ylim(0, 1)
        ax.grid(True)
        plt.title(title, fontsize=14, fontweight='bold', y=1.1)
        
        # Save the chart if output path is provided
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            return output_path
        else:
            # Create a temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = "temp_charts"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = f"{temp_dir}/feature_radar_chart_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(temp_path, dpi=300)
            plt.close()
            return temp_path
    
    except Exception as e:
        logger.error(f"Failed to create feature radar chart: {e}")
        return ""

def generate_session_report(session_id: str, output_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive PDF report for a session.
    
    Args:
        session_id: Session identifier
        output_path: Optional output file path
        
    Returns:
        Path to saved report
    """
    try:
        # Initialize data storage
        storage = DepressionDataStorage()
        
        # Get session data
        session_data = storage.get_session_results(session_id)
        if session_data.empty:
            logger.error(f"No data found for session {session_id}")
            return ""
        
        # Get session summary
        summary = storage.get_session_summary(session_id)
        
        # Create charts directory
        report_dir = "reports"
        charts_dir = f"{report_dir}/charts"
        os.makedirs(charts_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create depression score chart
        scores = session_data['depression_score'].tolist()
        timestamps = session_data['timestamp'].tolist()
        score_chart = create_depression_score_chart(
            scores, 
            timestamps, 
            f"Depression Score Trend - Session {session_id[:8]}",
            f"{charts_dir}/score_chart_{session_id[:8]}_{timestamp}.png"
        )
        
        # Set default output path if not provided
        if not output_path:
            output_path = f"{report_dir}/session_report_{session_id[:8]}_{timestamp}.html"
        
        # Create simple HTML report
        with open(output_path, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Depression Analysis Report - Session {session_id[:8]}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .section {{ margin-bottom: 30px; }}
        .score-chart {{ text-align: center; margin: 20px 0; }}
        .score-chart img {{ max-width: 100%; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .summary-box {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .risk-low {{ color: green; }}
        .risk-mild {{ color: #ffc107; }}
        .risk-moderate {{ color: orange; }}
        .risk-high {{ color: red; }}
        .risk-severe {{ color: purple; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Depression Analysis Report</h1>
        <div class="section summary-box">
            <h2>Session Summary</h2>
            <p><strong>Session ID:</strong> {session_id}</p>
            <p><strong>Date:</strong> {summary.get('timestamp', 'N/A')}</p>
            <p><strong>Total Samples:</strong> {summary.get('total_samples', 0)}</p>
            <p><strong>Average Depression Score:</strong> {summary.get('average_depression_score', 0):.1f}/100</p>
            <p><strong>Maximum Depression Score:</strong> {summary.get('max_depression_score', 0):.1f}/100</p>
        </div>
        
        <div class="section">
            <h2>Depression Score Trend</h2>
            <div class="score-chart">
                <img src="{os.path.relpath(score_chart, os.path.dirname(output_path))}" alt="Depression Score Trend">
            </div>
            <p>The chart above shows depression score variation throughout the session.</p>
        </div>
        
        <div class="section">
            <h2>Depression Level Distribution</h2>
            <table>
                <tr>
                    <th>Risk Level</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            """)
            
            # Add depression level distribution
            level_counts = summary.get('depression_level_distribution', {})
            total = sum(level_counts.values())
            for level, count in level_counts.items():
                percentage = (count / total * 100) if total > 0 else 0
                f.write(f"""
                <tr>
                    <td class="risk-{level.split()[0].lower()}">{level.upper()}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
                """)
            
            f.write("""
            </table>
        </div>
        
        <div class="section">
            <h2>Speech Analysis Samples</h2>
            <table>
                <tr>
                    <th>Timestamp</th>
                    <th>Transcript</th>
                    <th>Depression Score</th>
                    <th>Risk Level</th>
                </tr>
            """)
            
            # Add sample transcripts (up to 10 samples)
            samples = min(10, len(session_data))
            sample_indices = np.linspace(0, len(session_data) - 1, samples, dtype=int)
            
            for idx in sample_indices:
                row = session_data.iloc[idx]
                transcript = row['transcript']
                if len(transcript) > 100:
                    transcript = transcript[:100] + "..."
                    
                score = row['depression_score']
                level = row['depression_level']
                timestamp = row['timestamp']
                
                f.write(f"""
                <tr>
                    <td>{timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp}</td>
                    <td>{transcript}</td>
                    <td>{score:.1f}</td>
                    <td class="risk-{level.split()[0].lower()}">{level.upper()}</td>
                </tr>
                """)
            
            f.write("""
            </table>
        </div>
        
        <div class="section">
            <h2>Key Metrics</h2>
            """)
            
            # Add key metrics
            key_metrics = summary.get('key_metrics', {})
            if key_metrics:
                f.write("""
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Average Value</th>
                        <th>Interpretation</th>
                    </tr>
                """)
                
                metrics_info = [
                    ("average_negative_sentiment", "Negative Sentiment", "Higher values indicate more negative emotional tone"),
                    ("average_depression_keyword_ratio", "Depression Keywords", "Higher values indicate more depression-related vocabulary"),
                    ("average_self_focus", "Self-Focus", "Higher values indicate increased focus on self in speech")
                ]
                
                for key, name, interpretation in metrics_info:
                    if key in key_metrics:
                        value = key_metrics[key]
                        f.write(f"""
                        <tr>
                            <td>{name}</td>
                            <td>{value:.3f}</td>
                            <td>{interpretation}</td>
                        </tr>
                        """)
                
                f.write("</table>")
            
            f.write("""
        </div>
        
        <div class="section">
            <h3>Disclaimer</h3>
            <p>This analysis is for informational purposes only and is not a clinical diagnosis. 
            Always consult with qualified healthcare professionals for mental health concerns.</p>
            <p><small>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small></p>
        </div>
    </div>
</body>
</html>
            """)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Failed to generate session report: {e}")
        return ""
