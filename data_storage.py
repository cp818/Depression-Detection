import json
import os
import sqlite3
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class DepressionDataStorage:
    """Storage class for depression analysis data."""
    
    def __init__(self, db_path: str = "data/depression_data.db"):
        """
        Initialize the data storage.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()
        
    def _ensure_db_dir(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            timestamp TEXT,
            session_duration INTEGER,
            total_samples INTEGER
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            transcript TEXT,
            depression_score REAL,
            depression_level TEXT,
            sentiment_neg REAL,
            sentiment_pos REAL,
            sentiment_neu REAL,
            depression_keyword_ratio REAL,
            first_person_ratio REAL,
            word_count INTEGER,
            word_variety_ratio REAL,
            pause_ratio REAL,
            raw_features TEXT,
            FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def _get_connection(self):
        """Get SQLite connection."""
        return sqlite3.connect(self.db_path)
    
    def create_session(self, session_id: str) -> bool:
        """
        Create a new analysis session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO analysis_sessions (session_id, timestamp, session_duration, total_samples) VALUES (?, ?, ?, ?)",
                (session_id, timestamp, 0, 0)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False
    
    def update_session(self, session_id: str, duration: int, total_samples: int) -> bool:
        """
        Update session information.
        
        Args:
            session_id: Session identifier
            duration: Session duration in seconds
            total_samples: Total number of samples analyzed
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE analysis_sessions SET session_duration = ?, total_samples = ? WHERE session_id = ?",
                (duration, total_samples, session_id)
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False
    
    def save_analysis_result(self, 
                           session_id: str, 
                           transcript: str, 
                           depression_score: float, 
                           depression_level: str, 
                           features: Dict[str, Any]) -> bool:
        """
        Save an analysis result.
        
        Args:
            session_id: Session identifier
            transcript: Speech transcript
            depression_score: Depression score (0-100)
            depression_level: Depression level description
            features: Dictionary of extracted features
            
        Returns:
            Boolean indicating success
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            sentiment = features.get('sentiment', {})
            
            cursor.execute(
                """
                INSERT INTO analysis_results (
                    session_id, timestamp, transcript, depression_score, depression_level,
                    sentiment_neg, sentiment_pos, sentiment_neu,
                    depression_keyword_ratio, first_person_ratio, word_count,
                    word_variety_ratio, pause_ratio, raw_features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, timestamp, transcript, depression_score, depression_level,
                    sentiment.get('neg', 0.0), sentiment.get('pos', 0.0), sentiment.get('neu', 0.0),
                    features.get('depression_keyword_ratio', 0.0),
                    features.get('first_person_ratio', 0.0),
                    features.get('word_count', 0),
                    features.get('word_variety_ratio', 0.0),
                    features.get('pause_ratio', 0.0),
                    json.dumps(features)
                )
            )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to save analysis result: {e}")
            return False
    
    def get_session_results(self, session_id: str) -> pd.DataFrame:
        """
        Get all results for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            DataFrame of session results
        """
        try:
            conn = self._get_connection()
            query = "SELECT * FROM analysis_results WHERE session_id = ? ORDER BY timestamp"
            df = pd.read_sql_query(query, conn, params=(session_id,))
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Failed to get session results: {e}")
            return pd.DataFrame()
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Generate a summary of the session results.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session summary
        """
        df = self.get_session_results(session_id)
        
        if df.empty:
            return {"error": "No data found for session"}
        
        try:
            # Get session info
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM analysis_sessions WHERE session_id = ?", (session_id,))
            session_data = cursor.fetchone()
            conn.close()
            
            if not session_data:
                return {"error": "Session not found"}
            
            # Calculate summary statistics
            avg_depression_score = df['depression_score'].mean()
            max_depression_score = df['depression_score'].max()
            score_trend = list(df['depression_score'])
            
            # Count depression levels
            level_counts = df['depression_level'].value_counts().to_dict()
            
            # Summarize most significant features
            avg_neg_sentiment = df['sentiment_neg'].mean()
            avg_keyword_ratio = df['depression_keyword_ratio'].mean()
            avg_self_focus = df['first_person_ratio'].mean()
            
            return {
                "session_id": session_id,
                "timestamp": session_data[2],
                "duration": session_data[3],
                "total_samples": session_data[4],
                "average_depression_score": float(avg_depression_score),
                "max_depression_score": float(max_depression_score),
                "depression_level_distribution": level_counts,
                "score_trend": score_trend,
                "key_metrics": {
                    "average_negative_sentiment": float(avg_neg_sentiment),
                    "average_depression_keyword_ratio": float(avg_keyword_ratio),
                    "average_self_focus": float(avg_self_focus)
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")
            return {"error": str(e)}
    
    def export_session_data(self, session_id: str, format: str = "csv") -> str:
        """
        Export session data to a file.
        
        Args:
            session_id: Session identifier
            format: Export format ("csv" or "json")
            
        Returns:
            Path to exported file
        """
        df = self.get_session_results(session_id)
        
        if df.empty:
            return ""
        
        try:
            # Create export directory
            export_dir = "exports"
            os.makedirs(export_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{export_dir}/session_{session_id}_{timestamp}"
            
            if format.lower() == "csv":
                export_path = f"{filename}.csv"
                df.to_csv(export_path, index=False)
            elif format.lower() == "json":
                export_path = f"{filename}.json"
                df.to_json(export_path, orient="records")
            else:
                logger.error(f"Unsupported export format: {format}")
                return ""
            
            return export_path
        except Exception as e:
            logger.error(f"Failed to export session data: {e}")
            return ""
