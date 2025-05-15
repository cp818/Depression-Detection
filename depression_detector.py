import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, Tuple, List, Any

# Download NLTK resources (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

class DepressionDetector:
    """
    Class for detecting depression biomarkers in speech text.
    Uses linguistic, semantic, and sentiment features to assess depression risk.
    """
    
    def __init__(self):
        """Initialize the DepressionDetector with required resources."""
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Depression-related keywords
        self.depression_keywords = [
            'sad', 'lonely', 'depressed', 'hopeless', 'tired', 'exhausted',
            'worthless', 'guilty', 'empty', 'numb', 'pain', 'hurt', 'crying',
            'suicide', 'die', 'death', 'alone', 'darkness', 'useless', 'failure',
            'miserable', 'anxious', 'worried', 'struggle', 'suffering', 'unhappy',
            'desperate', 'helpless', 'pointless', 'meaningless', 'burden', 'lost'
        ]
        
        # First-person pronouns (indicators of self-focus)
        self.first_person_pronouns = ['i', 'me', 'my', 'mine', 'myself']
        
        # Feature weights (can be adjusted based on clinical data)
        self.weights = {
            'negative_sentiment': 2.5,
            'depression_keywords': 2.0,
            'first_person_focus': 1.0,
            'speech_rate': 1.5,
            'word_variety': 1.0,
            'pause_frequency': 1.0
        }
    
    def analyze_text(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze speech text for depression biomarkers.
        
        Args:
            text: The transcribed speech text
            
        Returns:
            A tuple containing:
            - Depression score (0-100, higher means more likely depressed)
            - Dictionary of extracted features
        """
        # Normalize text
        text = text.lower().strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        words = [w for w in tokens if w.isalpha()]
        
        # Calculate features
        features = {}
        
        # 1. Sentiment analysis
        sentiment = self.sia.polarity_scores(text)
        features['sentiment'] = sentiment
        
        # 2. Depression keyword frequency
        depression_word_count = sum(1 for word in words if word in self.depression_keywords)
        depression_keyword_ratio = depression_word_count / max(len(words), 1)
        features['depression_keyword_ratio'] = depression_keyword_ratio
        features['depression_keywords_found'] = [word for word in words if word in self.depression_keywords]
        
        # 3. First-person pronoun usage (self-focus)
        fp_count = sum(1 for word in words if word in self.first_person_pronouns)
        fp_ratio = fp_count / max(len(words), 1)
        features['first_person_ratio'] = fp_ratio
        
        # 4. Speech characteristics
        # Word count and speech rate (proxy)
        features['word_count'] = len(words)
        
        # 5. Word variety (lexical diversity)
        unique_words = len(set(words))
        word_variety_ratio = unique_words / max(len(words), 1)
        features['word_variety_ratio'] = word_variety_ratio
        
        # 6. Approximate pause frequency using punctuation as proxy
        pause_markers = ['.', ',', '...', ';', '—']
        pause_count = sum(1 for token in tokens if token in pause_markers)
        pause_ratio = pause_count / max(len(tokens), 1)
        features['pause_ratio'] = pause_ratio
        
        # Calculate depression score (0-100)
        score = 0
        score += self.weights['negative_sentiment'] * (sentiment['neg'] * 100)
        score += self.weights['depression_keywords'] * (depression_keyword_ratio * 100)
        score += self.weights['first_person_focus'] * (fp_ratio * 50)  # Less weight for self-focus
        
        # Lower speech rate may indicate depression
        speech_rate_factor = max(0, 1 - (len(words) / 150)) if len(text) > 50 else 0
        score += self.weights['speech_rate'] * (speech_rate_factor * 50)
        
        # Lower word variety may indicate depression
        word_variety_factor = max(0, 1 - word_variety_ratio)
        score += self.weights['word_variety'] * (word_variety_factor * 50)
        
        # Higher pause frequency may indicate depression
        score += self.weights['pause_frequency'] * (pause_ratio * 50)
        
        # Normalize score to 0-100 range
        normalized_score = min(100, max(0, score))
        
        return normalized_score, features
    
    def get_depression_level(self, score: float) -> str:
        """
        Convert numerical depression score to a descriptive level.
        
        Args:
            score: Depression score (0-100)
            
        Returns:
            String description of depression level
        """
        if score < 20:
            return "low risk"
        elif score < 40:
            return "mild risk"
        elif score < 60:
            return "moderate risk"
        elif score < 80:
            return "high risk"
        else:
            return "severe risk"
    
    def get_feedback(self, score: float, features: Dict[str, Any]) -> str:
        """
        Generate feedback based on depression analysis.
        
        Args:
            score: Depression score (0-100)
            features: Dictionary of extracted features
            
        Returns:
            String feedback with observations and recommendations
        """
        level = self.get_depression_level(score)
        
        feedback = f"Depression risk level: {level.upper()} ({score:.1f}/100)\n\n"
        
        # Add observations
        feedback += "Observations:\n"
        
        # Sentiment
        sentiment = features['sentiment']
        if sentiment['neg'] > 0.3:
            feedback += "- High negative emotional content detected in speech\n"
        
        # Depression keywords
        if features['depression_keywords_found']:
            feedback += f"- Depression-related keywords detected: {', '.join(features['depression_keywords_found'][:5])}\n"
            if len(features['depression_keywords_found']) > 5:
                feedback += f"  (and {len(features['depression_keywords_found']) - 5} more)\n"
        
        # First-person focus
        if features['first_person_ratio'] > 0.15:
            feedback += "- High self-focus in speech patterns\n"
        
        # Recommendations based on risk level
        feedback += "\nRecommendations:\n"
        if score >= 60:
            feedback += "- Consider consulting a mental health professional\n"
            feedback += "- This tool is not diagnostic but suggests potential concern\n"
        elif score >= 40:
            feedback += "- Consider monitoring mood patterns\n"
            feedback += "- Practice self-care activities\n"
        else:
            feedback += "- Continue monitoring for any significant changes\n"
        
        feedback += "\nNote: This is an automated analysis and not a clinical diagnosis."
        
        return feedback
