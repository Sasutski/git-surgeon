"""
Intent classifier for git-surgeon natural language commands.

This module uses a trained machine learning model to classify
user input into specific Git operation intents.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import re
from difflib import SequenceMatcher
from collections import defaultdict, OrderedDict
import time

# Add the project root directory to the Python path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from surgeon.intents import Intent, INTENT_SCHEMAS

class IntentClassifier:
    """
    Classifies natural language text into Git operation intents.
    Uses a trained model to predict the most likely operation the user intends.
    """
    
    # Intent-specific keyword patterns for higher confidence matching
    INTENT_PATTERNS = {
        'SQUASH': [
            r'(squash|combin|merg|collaps).+(commit|commits|together)',
            r'(make|turn).+into one commit',
            r'(join|unit).+commit',
            r'condens.+commit',
            r'flatten.+commit',
            r'consolidat.+commit',
        ],
        'REWORD': [
            r'(chang|edit|modif|updat).+(message|msg|commit message)',
            r'reword.+(commit|message)',
            r'(fix|correct).+commit message',
            r'rephras.+commit',
            r'(chang|edit).+description',
        ],
        'DROP': [
            r'(drop|delet|remov).+(commit|commits)',
            r'get rid of.+commit',
            r'(discard|eliminate).+(commit|commits)',
            r'(clear|purge).+(commit|commits)',
            r'(erased|remov).+(commit|commits)',
            # Add patterns that more clearly distinguish DROP from REORDER
            r'(delet|remov|drop).+\d+.+(commit|commits)',
            r'(delet|remov|drop).+(last|previous|recent).+\d*.+(commit|commits)',
        ],
        'REORDER': [
            r'(reorder|chang.+order|swap|rearrang).+(commit|commits)',
            r'(mov|switch).+commit.+(befor|after)',
            r'(revers|chang).+sequence',
            r'(rearrang|reposition).+(commit|order)',
        ],
        'SPLIT': [
            r'(split|break|divid|separat).+(commit|commits)',
            r'(mak|creat).+multiple commit',
            r'(divid|break.+down).+commit'
        ],
        'AMEND': [
            r'amend.+(commit|change)',
            r'(add|includ).+to.+(last|previous).+commit',
            r'(includ|fix).+(file|chang).+commit'
        ],
        'UNDO': [
            r'(undo|revert|rollback|reset|cancel).+(commit|change)',
            r'go.+back.+(before|to).+commit',
            r'(restor|return).+(before|previous)'
        ],
        'RENAME_BRANCH': [
            r'(renam|chang.+name).+branch',
            r'(call|name).+branch.+(to|as)',
            r'(set|updat|modif).+branch.+name'
        ],
        'REBASE_ONTO': [
            r'(rebas|mov|appl|transplant).+(onto|on top of|to).+(branch)',
            r'(mov|transplant|replay).+(commit|chang).+(onto|on top of)',
            r'rebas.+(branch|commit)'
        ]
    }
    
    # Action-target pattern matching for more accurate intent classification
    # Maps specific action words to likely intents for stronger disambiguation
    ACTION_INTENT_MAP = {
        # Strong DROP indicators
        'delete': 'DROP',
        'remove': 'DROP',
        'eliminate': 'DROP',
        'discard': 'DROP',
        'drop': 'DROP',
        
        # Strong SQUASH indicators
        'squash': 'SQUASH',
        'combine': 'SQUASH',
        'merge': 'SQUASH',
        'join': 'SQUASH',
        'flatten': 'SQUASH',
        
        # Strong REWORD indicators
        'change message': 'REWORD',
        'edit message': 'REWORD',
        'reword': 'REWORD',
        'rename message': 'REWORD',
        'update message': 'REWORD',
        
        # Strong REORDER indicators
        'reorder': 'REORDER',
        'swap': 'REORDER',
        'rearrange': 'REORDER',
        'move': 'REORDER',
        
        # Strong RENAME_BRANCH indicators
        'rename branch': 'RENAME_BRANCH',
        'change branch name': 'RENAME_BRANCH',
    }
    
    # Common git and relevant domain words for noise detection
    DOMAIN_WORDS = {
        'git', 'commit', 'commits', 'branch', 'branches', 'merge', 'rebase', 'squash', 'amend',
        'message', 'history', 'log', 'repo', 'repository', 'stash', 
        'push', 'pull', 'fetch', 'checkout', 'undo', 'delete', 'drop',
        'reword', 'edit', 'change', 'update', 'rename', 'split', 'combine',
        'fix', 'reset', 'revert', 'head', 'master', 'main', 'feature', 'develop', 
        'last', 'previous', 'first', 'recent', 'latest'
    }
    
    # Common misspellings of domain words with their corrections
    COMMON_TYPOS = {
        'mrge': 'merge',
        'mereg': 'merge',
        'megre': 'merge',
        'merg': 'merge',
        'squah': 'squash',
        'squahs': 'squash',
        'comit': 'commit',
        'committ': 'commit',
        'comitt': 'commit',
        'rebas': 'rebase',
        'rebace': 'rebase',
        'brach': 'branch',
        'branc': 'branch',
        'brnach': 'branch',
        'chang': 'change',
        'mesage': 'message',
        'messg': 'message',
        'messag': 'message',
        'rewrd': 'reword',
        'rword': 'reword',
        'delet': 'delete',
        'remov': 'remove',
        'undoo': 'undo'
    }
    
    # Minimum confidence threshold for a valid intent
    CONFIDENCE_THRESHOLD = 0.25
    
    # Fuzzy matching settings - adaptive based on word length
    # Shorter words require higher similarity to prevent false positives
    FUZZY_SHORT_WORD_LENGTH = 5  # Words shorter than this are considered "short"
    FUZZY_SHORT_THRESHOLD = 0.85  # Higher similarity required for short words
    FUZZY_NORMAL_THRESHOLD = 0.78  # Normal threshold for longer words
    
    # Caching configuration
    MAX_CACHE_SIZE = 100  # Maximum number of entries in cache
    CACHE_EXPIRY_SECONDS = 3600  # Cache entries expire after 1 hour
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier with a trained model.
        
        Args:
            model_path: Path to the saved model file. If None, uses the default path.
        """
        # Initialize prediction cache
        self._prediction_cache = OrderedDict()
        self._normalized_text_cache = OrderedDict()
        
        if model_path is None:
            # Default to the models directory relative to project root
            model_path = os.path.join(project_root, "models", "intent_classifier.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at: {model_path}")
        
        # Load the model
        model_data = joblib.load(model_path)
        
        # Handle both direct model loading and dict-style model info
        if isinstance(model_data, dict) and 'model' in model_data:
            self.model = model_data['model']
            self.label_encoder = model_data.get('label_encoder', None)
            self.metrics = model_data.get('metrics', {})
            self.feature_names = model_data.get('feature_names', None)
            self.input_features = self._get_required_features()
            self.class_names = model_data.get('class_names', [])
            
            # Log model metrics if available
            if self.metrics and 'accuracy' in self.metrics:
                print(f"Model loaded with accuracy: {self.metrics['accuracy']:.4f}")
        else:
            # Direct model object
            self.model = model_data
            self.label_encoder = None
            self.metrics = {}
            self.feature_names = None
            self.input_features = []
            self.class_names = []

        # Compile regex patterns for faster matching
        self.intent_regexes = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        
        # Build a length-keyed dictionary for more efficient fuzzy matching
        # Group domain words by length to reduce the search space
        self.domain_words_by_length = defaultdict(list)
        for word in self.DOMAIN_WORDS:
            self.domain_words_by_length[len(word)].append(word)

        # Pre-compile action patterns for faster matching
        self.action_patterns = {}
        for action, intent in self.ACTION_INTENT_MAP.items():
            # Create regex pattern for each action
            # Using word boundary to ensure we match whole words/phrases
            pattern = r'\b' + action.replace(' ', r'\s+') + r'\b'
            self.action_patterns[re.compile(pattern, re.IGNORECASE)] = intent

    def _get_required_features(self) -> List[str]:
        """
        Determine the required features for the model by inspecting its components.
        
        Returns:
            List of feature column names
        """
        # Try to extract feature names by looking at model components
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        elif hasattr(self.model, 'named_steps') and 'features' in self.model.named_steps:
            # For pipelines with ColumnTransformer
            features = []
            # Always include 'text' as this is our main feature
            features.append('text')
            # Add standard engineering features
            features.extend(['word_count', 'has_number', 
                           'has_squash_keyword', 'has_reword_keyword',
                           'has_branch_keyword', 'has_rebase_keyword'])
            return features
        else:
            # Default set of features
            return ['text']

    def _fuzzy_match_word(self, word: str) -> Tuple[bool, str]:
        """
        Check if a word closely matches any domain word using a more efficient approach.
        
        Args:
            word: Word to check
            
        Returns:
            Tuple of (is_match, matched_word)
        """
        # Skip very short words that would match too many things
        if len(word) < 3:
            return False, ""
            
        # Direct match
        if word in self.DOMAIN_WORDS:
            return True, word
            
        # Check known typos dictionary first (fastest lookup)
        if word in self.COMMON_TYPOS:
            return True, self.COMMON_TYPOS[word]
            
        # For longer words, apply progressive fuzzy matching
        if len(word) >= 4:
            # Only check against words of similar length to improve performance
            # This drastically reduces the number of comparisons
            word_len = len(word)
            
            # Determine appropriate similarity threshold based on word length
            threshold = (self.FUZZY_SHORT_THRESHOLD if word_len < self.FUZZY_SHORT_WORD_LENGTH 
                       else self.FUZZY_NORMAL_THRESHOLD)
            
            # Check only words with similar lengths (±2 characters)
            for check_len in range(word_len - 1, word_len + 2):
                if check_len < 3:
                    continue
                
                for target in self.domain_words_by_length[check_len]:
                    # Fast check: if the first and last character don't match, likely not similar
                    # This is a fast initial filter before doing expensive similarity calculation
                    if len(target) > 2 and len(word) > 2:
                        if word[0] != target[0] and word[-1] != target[-1]:
                            continue
                    
                    # Calculate similarity ratio
                    similarity = SequenceMatcher(None, word, target).ratio()
                    if similarity >= threshold:
                        return True, target
                    
        return False, ""

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by correcting common typos in domain words.
        Uses a smarter, more efficient approach to fuzzy matching.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text with typos corrected
        """
        if not text:
            return ""
            
        # Check cache first
        cache_key = text.lower().strip()
        if cache_key in self._normalized_text_cache:
            # Move to end to mark as recently used
            normalized = self._normalized_text_cache.pop(cache_key)
            self._normalized_text_cache[cache_key] = normalized
            return normalized
            
        # Process text if not in cache
        text_lower = text.lower()
        
        # Extract whole words for comparison
        words = re.findall(r'\b\w+\b', text_lower)
        if not words:
            return text_lower
        
        # Create a dictionary of replacements to apply
        replacements = {}
        
        # Check each word for possible corrections
        for word in words:
            # Skip small words and words we already know how to handle
            if len(word) <= 2 or word in replacements:
                continue
            
            # Handle special case for plural forms - don't "correct" them to singular
            # This ensures we don't incorrectly change "commits" to "commit"
            if word.endswith('s') and word[:-1] in self.DOMAIN_WORDS:
                continue
                
            # Only try to match words that could plausibly be git-related
            # This is a heuristic filter to avoid unnecessary fuzzy matching
            # E.g., words like "the", "and", "with" don't need fuzzy matching
            if any(char in 'abcdefghijklmnopqrstuvwxyz' for char in word):
                matched, correction = self._fuzzy_match_word(word)
                if matched and correction != word:  # Only need to replace if it's actually different
                    # Check if we're dealing with a potential plural
                    if word.endswith('s') and not correction.endswith('s'):
                        # If the original word is plural, keep it plural
                        correction = correction + 's'
                    replacements[word] = correction
        
        # Apply all replacements at once
        if replacements:
            # Sort by length descending to avoid partial word replacements
            for word, correction in sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True):
                text_lower = re.sub(r'\b' + re.escape(word) + r'\b', correction, text_lower)
        
        # Add result to cache
        self._normalized_text_cache[cache_key] = text_lower
        
        # Maintain cache size limit
        if len(self._normalized_text_cache) > self.MAX_CACHE_SIZE:
            # Remove oldest entry (first item in OrderedDict)
            self._normalized_text_cache.popitem(last=False)
        
        return text_lower

    def _prepare_features(self, text: str) -> pd.DataFrame:
        """
        Prepare text for prediction by transforming it into the expected feature format.
        
        Args:
            text: The natural language input text
            
        Returns:
            DataFrame with the properly formatted features
        """
        # Basic preprocessing to match training data
        text_lower = text.lower().strip()
        
        # Create a DataFrame with all expected features
        df = pd.DataFrame({'text': [text_lower]})
        
        # Add engineered features
        words = re.findall(r'\b\w+\b', text_lower)
        df['word_count'] = len(words)
        df['has_number'] = int(any(c.isdigit() for c in text_lower))
        
        # Define keyword groups for feature extraction with their stemmed versions
        squash_keywords = ['squash', 'combin', 'merg', 'collaps', 'join']
        reword_keywords = ['reword', 'messag', 'edit', 'chang', 'fix']
        branch_keywords = ['branch', 'name', 'renam']
        rebase_keywords = ['rebas', 'onto', 'top of']
        
        # Look for keywords in the text (including potential misspellings)
        has_squash_kw = any(kw in text_lower for kw in squash_keywords)
        has_reword_kw = any(kw in text_lower for kw in reword_keywords)
        has_branch_kw = any(kw in text_lower for kw in branch_keywords)
        has_rebase_kw = any(kw in text_lower for kw in rebase_keywords)
        
        # If exact matches not found, check for fuzzy matches in full words
        if not (has_squash_kw or has_reword_kw or has_branch_kw or has_rebase_kw):
            for word in words:
                matched, correction = self._fuzzy_match_word(word)
                if matched:
                    # Categorize the matched word
                    if correction in squash_keywords or any(kw in correction for kw in squash_keywords):
                        has_squash_kw = True
                    elif correction in reword_keywords or any(kw in correction for kw in reword_keywords):
                        has_reword_kw = True
                    elif correction in branch_keywords or any(kw in correction for kw in branch_keywords):
                        has_branch_kw = True
                    elif correction in rebase_keywords or any(kw in correction for kw in rebase_keywords):
                        has_rebase_kw = True
        
        # Set feature flags
        df['has_squash_keyword'] = int(has_squash_kw)
        df['has_reword_keyword'] = int(has_reword_kw)
        df['has_branch_keyword'] = int(has_branch_kw)
        df['has_rebase_keyword'] = int(has_rebase_kw)
            
        return df

    def _detect_primary_action(self, text: str) -> Tuple[str, float]:
        """
        Detect the primary action in the text to help disambiguate between similar intents.
        This serves as an additional signal beyond the ML model's prediction
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (intent_name, confidence_boost)
        """
        text_lower = text.lower()
        
        # Check for REORDER-specific patterns first - strengthen the pattern and boost
        if re.search(r'(switch|swap|exchang|revers|flip).+(position|order|place|commit)', text_lower):
            return 'REORDER', 0.75  # Much stronger confidence boost for explicit reordering operations
        
        # Check for commit hash pattern which strongly indicates REORDER
        hash_pattern = re.findall(r'\b([a-f0-9]{4,40})\b', text_lower)
        if len(hash_pattern) >= 2 and ('switch' in text_lower or 'swap' in text_lower or 'reorder' in text_lower):
            return 'REORDER', 0.8  # Very high confidence for hash-based reordering
    
        # Check for direct action matches
        for pattern, intent in self.action_patterns.items():
            if pattern.search(text_lower):
                return intent, 0.3  # Strong confidence boost for direct action match
        
        # Check for numeric patterns that indicate specific intents
        number_commits_pattern = re.search(r'(last|previous|recent)\s+(\d+)\s+(commit|commits)', text_lower)
        if number_commits_pattern:
            # "last X commits" is common for many operations, check for specific actions
            if any(term in text_lower for term in ['delete', 'remove', 'drop']):
                return 'DROP', 0.35
            elif any(term in text_lower for term in ['squash', 'combine', 'merge']):
                return 'SQUASH', 0.35
            # Default to a weaker signal if just "last X commits" with no clear action
            return '', 0.0
            
        # Check for commit modification patterns
        if ('message' in text_lower and any(term in text_lower for term in ['change', 'edit', 'update'])):
            return 'REWORD', 0.25
            
        # No clear primary action detected
        return '', 0.0

    def _boost_confidence(self, text: str, intent: str, base_confidence: float) -> float:
        """
        Boost confidence for predictions based on pattern matching.
        
        Args:
            text: Input text
            intent: Predicted intent
            base_confidence: Base confidence from model
            
        Returns:
            Boosted confidence score
        """
        # If confidence is already high, don't modify it
        if base_confidence > 0.95:
            return base_confidence
            
        # Normalize text to handle typos
        normalized_text = self._normalize_text(text)
        
        # Check for primary action that might override ML prediction
        detected_intent, action_boost = self._detect_primary_action(normalized_text)
        
        # If we detected a strong action signal and it matches the predicted intent,
        # apply a significant confidence boost
        if detected_intent and detected_intent == intent:
            return min(base_confidence + action_boost, 0.98)
            
        # If we detected a strong action signal but it DOESN'T match the predicted intent,
        # this is a red flag - reduce confidence slightly unless it's very high
        if detected_intent and detected_intent != intent and base_confidence < 0.85:
            return max(base_confidence - 0.05, 0.0)
        
        # Check if text matches any regex patterns for this intent
        if intent in self.intent_regexes:
            for pattern in self.intent_regexes[intent]:
                if pattern.search(normalized_text):
                    # Boost confidence based on pattern match
                    boost_factor = 0.15
                    # Don't let confidence exceed 0.98
                    return min(base_confidence + boost_factor, 0.98)
        
        # Additional intent-specific confidence boosts
        if intent == 'SQUASH':
            # Key pattern: 'merge/combine/squash' + number + 'commits'
            if re.search(r'(merge|combin|squash|join).+\d+.*(commit|commits)', normalized_text):
                return min(base_confidence + 0.35, 0.98)
                
            # Another common pattern: merge/combine last/recent/previous commits
            if re.search(r'(merge|combin|squash|join).+(last|recent|previous).*(commit|commits)', normalized_text):
                return min(base_confidence + 0.30, 0.98)
                
            # Check both original and normalized text
            if any(kw in normalized_text for kw in ['squash', 'combine', 'merge']):
                return min(base_confidence + 0.12, 0.98)
                
        elif intent == 'DROP':
            # Key pattern: 'delete/remove/drop' + number + 'commits'
            if re.search(r'(delet|remov|drop).+\d+.*(commit|commits)', normalized_text):
                return min(base_confidence + 0.40, 0.98)
                
            # Another strong pattern: delete/remove/drop + last/recent/previous + commits
            if re.search(r'(delet|remov|drop).+(last|recent|previous).*(commit|commits)', normalized_text):
                return min(base_confidence + 0.40, 0.98)
                
        elif intent == 'REWORD':
            if any(kw in normalized_text for kw in ['message', 'commit message']):
                return min(base_confidence + 0.12, 0.98)
                
        elif intent == 'RENAME_BRANCH':
            if 'branch' in normalized_text and ('name' in normalized_text or 'rename' in normalized_text):
                return min(base_confidence + 0.15, 0.98)
            
        # No boost applied
        return base_confidence

    def clear_cache(self):
        """
        Clear all prediction caches. Use when memory usage is a concern
        or when updating the model.
        """
        self._prediction_cache.clear()
        self._normalized_text_cache.clear()
    
    def predict(self, text: str) -> str:
        """
        Predict the raw string label of the intent.
        
        Args:
            text: The natural language input text
            
        Returns:
            The predicted intent label as a string
        """
        # Check cache first
        cache_key = text.lower().strip()
        if cache_key in self._prediction_cache:
            cache_entry = self._prediction_cache.pop(cache_key)  # Remove and re-add to update LRU order
            # Check if the cache entry has expired
            if time.time() - cache_entry['timestamp'] <= self.CACHE_EXPIRY_SECONDS:
                # Move to end to mark as recently used
                self._prediction_cache[cache_key] = cache_entry
                return cache_entry['prediction']
        
        # Apply typo correction for prediction
        normalized_text = self._normalize_text(text)
        
        # Check for noise/random input
        if self._is_likely_noise(text):
            # Store in cache
            self._prediction_cache[cache_key] = {
                'prediction': "UNKNOWN", 
                'timestamp': time.time()
            }
            # Maintain cache size limit
            if len(self._prediction_cache) > self.MAX_CACHE_SIZE:
                self._prediction_cache.popitem(last=False)  # Remove oldest
            return "UNKNOWN"
            
        X = self._prepare_features(normalized_text)
        
        # Make prediction
        try:
            prediction = self.model.predict(X)[0]
            
            # Convert prediction to string if it's not already
            if isinstance(prediction, (np.ndarray, np.int64, np.integer, int)):
                # If we have a label encoder, use it to get the original label
                if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                    prediction = self.label_encoder.inverse_transform([prediction])[0]
                else:
                    prediction = str(prediction)
            
            # Store in cache
            self._prediction_cache[cache_key] = {
                'prediction': prediction, 
                'timestamp': time.time()
            }
            
            # Maintain cache size limit
            if len(self._prediction_cache) > self.MAX_CACHE_SIZE:
                self._prediction_cache.popitem(last=False)  # Remove oldest
                
            return prediction
        except Exception as e:
            raise ValueError(f"Error predicting intent: {str(e)}")

    def predict_enum(self, text: str) -> Intent:
        """
        Predict the intent and return it as an Intent enum.
        
        Args:
            text: The natural language input text
            
        Returns:
            The predicted intent as an Intent enum
        """
        label = self.predict(text)
        
        # Ensure label is a string before calling upper()
        if not isinstance(label, str):
            label = str(label)
            
        try:
            return Intent[label.upper()]
        except KeyError:
            # Handle unknown intents more gracefully
            if label.upper() == "UNKNOWN":
                # If we have UNKNOWN intent defined, use it
                if hasattr(Intent, "UNKNOWN"):
                    return Intent.UNKNOWN
                else:
                    # Otherwise, use a default intent
                    return Intent.DROP  # Replace with appropriate fallback
            raise ValueError(f"Unknown intent label: {label}")
    
    def predict_with_confidence(self, text: str) -> Tuple[str, Optional[float]]:
        """
        Predict the intent with confidence score.
        
        Args:
            text: The natural language input text
            
        Returns:
            A tuple of (intent_label, confidence_score)
        """
        # Check cache first
        cache_key = text.lower().strip()
        if cache_key in self._prediction_cache:
            cache_entry = self._prediction_cache.pop(cache_key)
            # Check if the cache entry has confidence and hasn't expired
            if 'confidence' in cache_entry and time.time() - cache_entry['timestamp'] <= self.CACHE_EXPIRY_SECONDS:
                # Move to end to mark as recently used
                self._prediction_cache[cache_key] = cache_entry
                return (cache_entry['prediction'], cache_entry['confidence'])
        
        # Apply typo correction for prediction
        normalized_text = self._normalize_text(text)
        
        X = self._prepare_features(normalized_text)
        
        if not hasattr(self.model, 'predict_proba'):
            # Fallback if model doesn't support probabilities
            prediction = self.predict(text)  # Use the updated predict method
            return (prediction, None)
        
        try:
            # Check for noise first
            is_noise = self._is_likely_noise(text)
            
            # Get probabilities for each class
            probabilities = self.model.predict_proba(X)[0]
            # Get the highest probability and its index
            max_prob_idx = probabilities.argmax()
            confidence = probabilities[max_prob_idx]
            
            # For likely noise, adjust confidence down
            if is_noise:
                confidence = self._adjust_confidence_for_noise(text, confidence)
                
            # Get the class label for the highest probability
            if hasattr(self.model, 'classes_'):
                label = self.model.classes_[max_prob_idx]
                # Convert numeric label to original string label if needed
                if hasattr(self, 'label_encoder') and self.label_encoder is not None and not isinstance(label, str):
                    label = self.label_encoder.inverse_transform([label])[0]
            else:
                # Try to get the prediction directly
                label = self.predict(text)
                
            # Check additional patterns to boost confidence
            if isinstance(label, str) and not is_noise:
                confidence = self._boost_confidence(text, label.upper(), confidence)
            
            # Return UNKNOWN if confidence is too low
            if confidence < self.CONFIDENCE_THRESHOLD:
                # Store in cache
                self._prediction_cache[cache_key] = {
                    'prediction': "UNKNOWN",
                    'confidence': confidence,
                    'timestamp': time.time()
                }
                # Maintain cache size
                if len(self._prediction_cache) > self.MAX_CACHE_SIZE:
                    self._prediction_cache.popitem(last=False)
                return ("UNKNOWN", confidence)
            
            # Store in cache
            self._prediction_cache[cache_key] = {
                'prediction': label,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            # Maintain cache size
            if len(self._prediction_cache) > self.MAX_CACHE_SIZE:
                self._prediction_cache.popitem(last=False)
                
            return (label, confidence)
        except Exception as e:
            raise ValueError(f"Error predicting intent with confidence: {str(e)}")
            
    def get_top_n_predictions(self, text: str, n: int = 3) -> List[Tuple[str, float]]:
        """
        Get the top N predictions with confidence scores.
        
        Args:
            text: The natural language input text
            n: Number of top predictions to return
            
        Returns:
            List of (intent, confidence) tuples, sorted by confidence
        """
        # Check cache first
        cache_key = f"{text.lower().strip()}:top{n}"
        if cache_key in self._prediction_cache:
            cache_entry = self._prediction_cache.pop(cache_key)
            # Check if the cache entry has top_predictions and hasn't expired
            if 'top_predictions' in cache_entry and time.time() - cache_entry['timestamp'] <= self.CACHE_EXPIRY_SECONDS:
                # Move to end to mark as recently used
                self._prediction_cache[cache_key] = cache_entry
                return cache_entry['top_predictions']
        
        # Apply typo correction for prediction
        normalized_text = self._normalize_text(text)
        
        # First check for specific high-priority patterns that need special handling
        
        # Check for commit hash references - these are DROP operations
        hash_match = re.search(r'(?:delete|remove|drop|discard).+(?:commit|sha)\s+([a-f0-9]{4,40})', normalized_text, re.IGNORECASE)
        if hash_match or re.search(r'(?:delete|remove|drop|discard).+\b([a-f0-9]{6,40})\b', normalized_text, re.IGNORECASE):
            # This is definitely a DROP intent
            results = [("DROP", 0.95)]
            for i in range(1, min(n, 3)):
                results.append(("UNKNOWN", 0.1))
            
            # Cache and return results
            self._prediction_cache[f"{text.lower().strip()}:top{n}"] = {
                'top_predictions': results,
                'timestamp': time.time()
            }
            
            return results
            
        # Definite DROP cases: when user clearly says to drop/delete/remove commits with a number
        drop_pattern = re.search(r'(drop|delet|remov|discard)\s+(?:the\s+)?(last|previous|recent)\s+(\d+)?\s*(commit|commits)?', normalized_text)
        if drop_pattern or re.search(r'(drop|delet|remov|discard).+(commit|commits)', normalized_text):
            # This is definitely a DROP intent with high confidence, don't even bother with the model
            # Do regular prediction but then override the results
            try:
                # Do normal prediction first
                results = self._get_model_predictions(normalized_text, text, n, is_noise=False)
                
                # Now override with DROP at high confidence
                drop_confidence = 0.98  # Very high confidence
                
                # Find if DROP is already in the results
                drop_idx = next((i for i, (label, _) in enumerate(results) if label == "DROP"), None)
                
                if drop_idx is not None:
                    # Replace the existing DROP entry with our high confidence one
                    results[drop_idx] = ("DROP", drop_confidence)
                else:
                    # Insert DROP at the beginning
                    results.insert(0, ("DROP", drop_confidence))
                    
                # Resort results
                results.sort(key=lambda x: x[1], reverse=True)
                # Trim to top N
                results = results[:n]
                
                # Cache and return results
                self._prediction_cache[f"{text.lower().strip()}:top{n}"] = {
                    'top_predictions': results,
                    'timestamp': time.time()
                }
                
                return results
            except Exception as e:
                # Fall through to normal prediction if this special case handling fails
                pass
        
        # Is the input likely noise?
        is_noise = self._is_likely_noise(text)
        
        # Use the regular prediction pipeline
        return self._get_model_predictions(normalized_text, text, n, is_noise)
    
    def _get_model_predictions(self, normalized_text: str, original_text: str, n: int, is_noise: bool) -> List[Tuple[str, float]]:
        """
        Helper method to get predictions from the model.
        This contains the core prediction logic extracted from get_top_n_predictions.
        
        Args:
            normalized_text: The normalized text
            original_text: The original user input
            n: Number of predictions to return
            is_noise: Whether the input is likely noise
            
        Returns:
            List of (intent, confidence) tuples sorted by confidence
        """
        # Prepare features
        X = self._prepare_features(normalized_text)
        
        try:
            # Get probabilities for all classes
            probabilities = self.model.predict_proba(X)[0]
            
            # Get indices of top N probabilities
            top_indices = np.argsort(probabilities)[::-1][:n]
            
            results = []
            for idx in top_indices:
                confidence = probabilities[idx]
                if hasattr(self.model, 'classes_'):
                    label = self.model.classes_[idx]
                    # Convert numeric label to original string label if needed
                    if hasattr(self, 'label_encoder') and self.label_encoder is not None and not isinstance(label, str):
                        label = self.label_encoder.inverse_transform([label])[0]
                        
                    # Apply confidence adjustment for noise
                    if is_noise:
                        confidence = self._adjust_confidence_for_noise(original_text, confidence)
                    # Otherwise apply confidence boosting for non-noise
                    elif isinstance(label, str):
                        confidence = self._boost_confidence(original_text, label.upper(), confidence)
                        
                    results.append((label, confidence))
            
            # If likely noise and confidence is low, prepend UNKNOWN
            if is_noise and (not results or results[0][1] < self.CONFIDENCE_THRESHOLD):
                # Insert UNKNOWN as first result
                uncertain_conf = min(results[0][1] if results else 0.2, 0.22)
                results = [("UNKNOWN", uncertain_conf)] + results[:min(len(results), n-1)]
            
            # Special handling for primary action detection - may override model's prediction
            if not is_noise:
                detected_intent, action_boost = self._detect_primary_action(normalized_text)
                
                # If we detected a strong action with high confidence
                if detected_intent and action_boost >= 0.3:
                    # Check if the detected intent is already in the results
                    detected_idx = next((i for i, (label, _) in enumerate(results) 
                                        if label.upper() == detected_intent), None)
                    
                    if detected_idx is not None:
                        # It's already in results, boost confidence significantly
                        label, conf = results[detected_idx]
                        # Apply significant boost since this is based on explicit action terms
                        new_conf = min(conf + action_boost, 0.98)
                        results[detected_idx] = (label, new_conf)
                        # Resort based on new confidences
                        results.sort(key=lambda x: x[1], reverse=True)
                    else:
                        # Detected intent not in top results, but action signal is strong enough
                        # to include it (only if confidence boost is high)
                        if action_boost >= 0.35:
                            # Add it to results
                            results.append((detected_intent, 0.8))
                            # Resort
                            results.sort(key=lambda x: x[1], reverse=True)
                            # Keep only top N
                            results = results[:n]
                
                # Special case for "drop/delete X commits" pattern - should strongly favor DROP over REORDER
                if re.search(r'(drop|delet|remov).+(\d+|last).+(commit|commits)', normalized_text):
                    # Check if both DROP and REORDER are in the results
                    drop_idx = next((i for i, (label, _) in enumerate(results) if label == "DROP"), None)
                    reorder_idx = next((i for i, (label, _) in enumerate(results) if label == "REORDER"), None)
                    
                    # Ensure DROP has very high confidence
                    if drop_idx is not None:
                        # Boost DROP to very high confidence
                        results[drop_idx] = ("DROP", 0.97)  # Nearly certain
                        
                    # If both are present, ensure DROP has higher confidence
                    if drop_idx is not None and reorder_idx is not None:
                        # Resort to ensure DROP comes first
                        results.sort(key=lambda x: x[1], reverse=True)
                        
                    # If only REORDER is present but text suggests DROP, add DROP
                    elif drop_idx is None and reorder_idx is not None:
                        results.insert(0, ("DROP", 0.95))  # Insert DROP at beginning with high confidence
                        # Keep only top N
                        results = results[:n]
            
            return results
        except Exception as e:
            raise ValueError(f"Error getting top predictions: {str(e)}")

    def cache_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the internal caches.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'prediction_cache_size': len(self._prediction_cache),
            'normalized_text_cache_size': len(self._normalized_text_cache),
            'max_cache_size': self.MAX_CACHE_SIZE,
            'cache_expiry_seconds': self.CACHE_EXPIRY_SECONDS
        }

    def extract_intent_parameters(self, text: str) -> Dict[str, Any]:
        """
        Extract parameters from the text based on predicted intent.
        Uses the schema defined in intents.py to extract relevant parameters.
        
        Args:
            text: The natural language input text
            
        Returns:
            Dictionary of extracted parameters
        """
        # Apply typo correction before parameter extraction
        normalized_text = self._normalize_text(text)
        
        # Get intent
        try:
            intent_enum = self.predict_enum(normalized_text)
            intent_name = intent_enum.name
        except ValueError:
            # If we can't determine intent, return empty params
            return {}
        
        params = {}
        
        # Get parameter schema from intents.py
        schema = INTENT_SCHEMAS.get(intent_enum, {})
        
        # Handle different schema formats
        if isinstance(schema.get("params"), dict):
            # New format: dictionary of parameters with metadata
            required_params = [param for param, metadata in schema.get("params", {}).items() 
                              if metadata.get("required", True)]
            optional_params = [param for param, metadata in schema.get("params", {}).items() 
                              if not metadata.get("required", True)]
            all_params = list(schema.get("params", {}).keys())
        else:
            # Old format: simple list of parameters
            required_params = schema.get("params", [])
            optional_params = []
            all_params = required_params
        
        # Extract parameters based on intent
        if intent_name == "DROP":
            # Message-based identification - handle first as special case
            msg_match = re.search(r'(with\s+message|message\s+is|says|containing)\s*[\'"]([^\'"]+)[\'"]', normalized_text, re.IGNORECASE)
            if not msg_match:
                msg_match = re.search(r'(with\s+message|message\s+is|says|containing)\s*"([^"]+)"', normalized_text, re.IGNORECASE)
            
            # For DROP intent, we need to be more careful with quoted text
            # Only use general quoted text if it clearly looks like a commit message reference
            # and not some other parameter
            if not msg_match:
                raw_quotes = re.search(r'[\'"](.*?)[\'"]', normalized_text)
                if raw_quotes and ('commit' in normalized_text):
                    # Check if the quote appears to be a message reference
                    # rather than some other parameter
                    quoted_text = raw_quotes.group(1)
                    before_quote = normalized_text[:raw_quotes.start()].strip()
                    if ('delete' in before_quote or 'drop' in before_quote or 'remove' in before_quote) and \
                       not any(kw in before_quote for kw in ['to', 'as', 'with name', 'message as']):
                        msg_match = raw_quotes
            
            if msg_match:
                # Store as message_match parameter rather than embedding in target
                message = msg_match.group(2) if len(msg_match.groups()) > 1 else msg_match.group(1)
                params["message_match"] = message
                # Search in a larger history window for message matching
                params["target"] = "HEAD~50..HEAD"
                
            # First handle the "delete the latest commit" case
            if re.search(r'(delete|remove|drop|discard).+(latest|most\s+recent)', normalized_text, re.IGNORECASE):
                params["target"] = "HEAD"
                return params
                
            # Handle "delete the last commit"
            if re.search(r'(delete|remove|drop|discard)\s+(?:the\s+)?(?:last|previous|recent)\s+commit', normalized_text, re.IGNORECASE):
                params["target"] = "HEAD"
                return params
            
            # Check for "drop/delete the last X commits" pattern - most common case
            last_n_match = re.search(r'(?:drop|delet|remov|discard)\s+(?:the\s+)?(?:last|previous|recent)\s+(\d+)\s+(?:commit|commits)', normalized_text, re.IGNORECASE)
            if last_n_match:
                count = int(last_n_match.group(1))
                if count > 1:
                    params["target"] = f"HEAD~{count-1}..HEAD"
                else:
                    params["target"] = "HEAD"
                return params
            
            # Look for commit hash pattern - direct hash reference
            hash_match = re.search(r'(?:commit|sha|hash|id)\s+([a-f0-9]{4,40})', normalized_text, re.IGNORECASE)
            if hash_match:
                params["target"] = hash_match.group(1)
                return params
                
            # Try to find hash without the 'commit' keyword
            bare_hash_match = re.search(r'\b([a-f0-9]{6,40})\b', normalized_text)
            if bare_hash_match:
                params["target"] = bare_hash_match.group(1)
                return params
            
            # Check for "last commit" or similar
            if re.search(r'last\s+commit|previous\s+commit|recent\s+commit', normalized_text, re.IGNORECASE):
                params["target"] = "HEAD"
                return params
            
            # Check for ordinal position references like "second last commit", "third last", etc.
            ordinal_match = re.search(r'(second|third|fourth|fifth|2nd|3rd|4th|5th)\s+(?:last|previous|recent)\s+commit', normalized_text, re.IGNORECASE)
            if ordinal_match:
                # Map ordinals to their numeric values
                ordinal_map = {
                    'second': 1, '2nd': 1,
                    'third': 2, '3rd': 2,
                    'fourth': 3, '4th': 3,
                    'fifth': 4, '5th': 4
                }
                ordinal = ordinal_match.group(1).lower()
                # For "second last commit", we want HEAD~1, etc.
                params["target"] = f"HEAD~{ordinal_map.get(ordinal, 1)}"
                return params
                
            # Alternative pattern for "last N commits"
            last_n_alt = re.search(r'(?:last|previous|recent)\s+(\d+)\s+(?:commit|commits)', normalized_text)
            if last_n_alt:
                # For multi-commit operations, we need to specify a range
                count = int(last_n_alt.group(1))
                if count > 1:  # For multiple commits
                    params["target"] = f"HEAD~{count-1}..HEAD"
                else:  # For single commit
                    params["target"] = "HEAD"
                return params
                    
            # Direct number pattern - "delete N commits" without "last"
            number_match = re.search(r'(?:delete|remove|drop|discard)\s+(\d+)\s+(?:commit|commits)', normalized_text)
            if number_match and "target" not in params:
                count = int(number_match.group(1))
                if count > 1:
                    params["target"] = f"HEAD~{count-1}..HEAD"
                else:
                    params["target"] = "HEAD"
                return params
                    
            # Check for numbered commit without context (like "commit 3")
            commit_num_match = re.search(r'commit\s+(\d+)', normalized_text)
            if commit_num_match and "target" not in params:
                commit_num = int(commit_num_match.group(1))
                # For a specific numbered commit, we want HEAD~(N-1)
                params["target"] = f"HEAD~{commit_num-1}"
                return params
                
            # Handle "delete from X to Y" pattern
            range_match = re.search(r'from\s+(\S+)\s+to\s+(\S+)', normalized_text)
            if range_match:
                params["target"] = f"{range_match.group(1)}..{range_match.group(2)}"
                return params
                
            # If we've exhausted all specific patterns and still have nothing,
            # check if the text clearly indicates deleting a commit
            if "message_match" not in params and re.search(r'(delete|remove|drop|discard).+commit', normalized_text, re.IGNORECASE):
                # Default to HEAD when the intent is clear but target is ambiguous
                params["target"] = "HEAD"
                return params
            
            # If we have a message_match but no target, set a default target
            # This allows the executor to decide how to handle message matching
            if "message_match" in params and "target" not in params:
                # Search in a larger history window (last 50 commits)
                params["target"] = "HEAD~50..HEAD"

        elif intent_name == "SQUASH":
            # Look for number of commits to squash
            number_match = re.search(r'(\d+)(?:\s+commit)', normalized_text)
            if not number_match:
                # Try alternative pattern with plural "commits"
                number_match = re.search(r'(\d+)(?:\s+commits)', normalized_text)
                
            if number_match:
                params["count"] = int(number_match.group(1))
                
            # Look for last/recent X commits pattern
            last_match = re.search(r'(last|previous|recent)\s+(\d+)', normalized_text)
            if last_match:
                params["count"] = int(last_match.group(2))
            
            # Look for most recent/last commit (singular)
            if "last commit" in normalized_text or "most recent commit" in normalized_text:
                if "count" not in params:
                    params["count"] = 1
                
            # Look for commit message
            msg_match = re.search(r'(message|name|msg|called|with)\s*[\'"]([^\'"]+)[\'"]', normalized_text, re.IGNORECASE)
            if not msg_match:
                # Try with double quotes
                msg_match = re.search(r'(message|name|msg|called|with)\s*"([^"]+)"', normalized_text, re.IGNORECASE)
            if not msg_match:
                # Try more general pattern to capture message without specific keywords
                msg_match = re.search(r'[\'"](.*?)[\'"]', normalized_text)
                
            if msg_match:
                params["new_message"] = msg_match.group(2) if len(msg_match.groups()) > 1 else msg_match.group(1)
        
        elif intent_name == "REWORD":
            # Look for target commit
            target_match = re.search(r'(commit)\s+([a-f0-9]{4,40})', normalized_text, re.IGNORECASE)
            if target_match:
                params["target"] = target_match.group(2)
                
            # Check for "last" or "previous" commit
            if "last commit" in normalized_text or "previous commit" in normalized_text or "most recent commit" in normalized_text:
                params["target"] = "HEAD"
                
            # Check for ordinal position references like "second last commit", "third last", etc.
            ordinal_match = re.search(r'(second|third|fourth|fifth|2nd|3rd|4th|5th)\s+(last|previous|recent)\s+commit', normalized_text, re.IGNORECASE)
            if ordinal_match:
                # Map ordinals to their numeric values
                ordinal_map = {
                    'second': 1, '2nd': 1,
                    'third': 2, '3rd': 2,
                    'fourth': 3, '4th': 3,
                    'fifth': 4, '5th': 4
                }
                ordinal = ordinal_match.group(1).lower()
                # For "second last commit", we want HEAD~1, etc.
                params["target"] = f"HEAD~{ordinal_map.get(ordinal, 1)}"
                
            # Look for commit message
            msg_match = re.search(r'(to|with|as)\s*[\'"]([^\'"]+)[\'"]', normalized_text, re.IGNORECASE)
            if not msg_match:
                msg_match = re.search(r'(message|name|msg|called|with)\s*[\'"]([^\'"]+)[\'"]', normalized_text, re.IGNORECASE)
            if not msg_match:
                # Try with double quotes
                msg_match = re.search(r'(message|name|msg|called|with)\s*"([^"]+)"', normalized_text, re.IGNORECASE)
            if not msg_match:
                # Try more general pattern to capture message without specific keywords
                msg_match = re.search(r'[\'"](.*?)[\'"]', normalized_text)
                
            if msg_match:
                params["new_message"] = msg_match.group(2) if len(msg_match.groups()) > 1 else msg_match.group(1)
                
            # Default to HEAD when no target is explicitly provided
            if "target" not in params:
                params["target"] = "HEAD"

        elif intent_name == "REORDER":
            # Handle "reorder the last N commits" pattern - this needs special handling
            last_n_commits = re.search(r'(?:reorder|switch|reverse|flip)\s+(?:the\s+)?(?:last|previous|recent)\s+(\d+)\s+(?:commit|commits)', normalized_text, re.IGNORECASE)
            if last_n_commits:
                count = int(last_n_commits.group(1))
                if count > 1:
                    # Instead of assuming "reverse" order, just set the target range
                    # and let the executor handle the ordering interactively
                    # Don't set an explicit "order" parameter for this case
                    params["target"] = f"HEAD~{count-1}..HEAD"
                    return params
    
            # Only set "order: reverse" for explicit reverse/flip commands
            reverse_pattern = re.search(r'(?:reverse|flip|backwards|opposite)\s+(?:the\s+)?(?:order\s+(?:of\s+)?)?', normalized_text, re.IGNORECASE)
            if reverse_pattern and "last" in normalized_text:
                # Extract the count if available
                count_match = re.search(r'(?:last|previous|recent)\s+(\d+)', normalized_text)
                count = int(count_match.group(1)) if count_match else 2
                
                # Now set both parameters
                params["order"] = "reverse" 
                params["target"] = f"HEAD~{count-1}..HEAD"
                return params
                
            # Look for specific commit hashes to switch
            hash_swap_match = re.search(r'(?:switch|swap|exchang|revers|flip).+(?:position|order|place).+(?:of\s+)?(?:commit|hash)?\s*[\'"]?([a-f0-9]{4,40})[\'"]?\s+(?:and|with)\s+[\'"]?([a-f0-9]{4,40})[\'"]?', normalized_text, re.IGNORECASE)
            if not hash_swap_match:
                # Try simpler pattern without surrounding text
                hash_swap_match = re.search(r'(?:commit|hash)?\s*[\'"]?([a-f0-9]{4,40})[\'"]?\s+(?:and|with)\s+[\'"]?([a-f0-9]{4,40})[\'"]?', normalized_text, re.IGNORECASE)
        
            if hash_swap_match:
                # Get the two commit hashes to swap
                first_hash = hash_swap_match.group(1)
                second_hash = hash_swap_match.group(2)
                params["order"] = [second_hash, first_hash]
                return params
            
            # Alternate pattern for swapping two hashes without "and" or "with"
            hash_direct_match = re.findall(r'\b([a-f0-9]{4,40})\b', normalized_text)
            if len(hash_direct_match) >= 2 and any(word in normalized_text for word in ['switch', 'swap', 'reorder', 'exchange', 'position']):
                # Extract the first two hashes in reverse order (since we're swapping them)
                params["order"] = [hash_direct_match[1], hash_direct_match[0]]
                return params
            
            # Look for order specification
            order_match = re.search(r'order\s*:\s*(.+?)(?:$|\.|;)', normalized_text)
            if order_match:
                order_text = order_match.group(1).strip()
                # Split by commas if present
                if ',' in order_text:
                    params["order"] = [s.strip() for s in order_text.split(',')]
                else:
                    # Split by spaces
                    params["order"] = order_text.split()
            
            # Look for "swap X and Y" pattern
            swap_match = re.search(r'swap\s+(.+?)\s+and\s+(.+?)(?:$|\.|;)', normalized_text)
            if swap_match:
                params["order"] = [swap_match.group(1), swap_match.group(2)]
                
            # Look for "reverse" pattern - this is a special case
            reverse_match = re.search(r'reverse\s+(?:the\s+)?(?:order\s+(?:of\s+)?)?(?:the\s+)?(?:last|previous|recent)?\s*(\d+)?\s*(?:commit|commits)?', normalized_text, re.IGNORECASE)
            if reverse_match:
                count = int(reverse_match.group(1)) if reverse_match.group(1) else 2
                # For a reverse operation, we specify "reverse" as the order and provide a target range
                params["order"] = "reverse"
                params["target"] = f"HEAD~{count-1}..HEAD"
                return params
                
            # If we still don't have order, try to derive from the context
            if "order" not in params:
                # Check for numeric patterns like "reorder commits 1, 2, 3" or "change order to 3, 1, 2"
                nums = re.findall(r'(\d+)', normalized_text)
                if nums:
                    if "reverse" in normalized_text or "backwards" in normalized_text or "opposite" in normalized_text:
                        # Handle reverse ordering case
                        if len(nums) == 1:
                            # If only one number, it's the count of commits to reverse
                            count = int(nums[0])
                            params["order"] = "reverse"
                            params["target"] = f"HEAD~{count-1}..HEAD"
                        else:
                            # If multiple numbers, they represent the desired order
                            params["order"] = nums
                    else:
                        params["order"] = nums
                # Look for commit references
                elif "last" in normalized_text or "head" in normalized_text:
                    # Check if it's a reverse request
                    if "reverse" in normalized_text or "backwards" in normalized_text or "opposite" in normalized_text:
                        # Default to reversing 2 commits if not specified
                        params["order"] = "reverse"
                        params["target"] = "HEAD~2..HEAD"
                    else:
                        # Try to get the number of commits to reorder
                        num_match = re.search(r'(?:last|recent)\s+(\d+)', normalized_text)
                        if num_match:
                            count = int(num_match.group(1))
                            if count > 2:  # For reordering multiple commits
                                params["order"] = "reverse"
                                params["target"] = f"HEAD~{count-1}..HEAD"
                            else:  # Default for 2 or fewer
                                params["order"] = ["HEAD~2", "HEAD~1", "HEAD"]
                        else:
                            # Default if no number specified
                            params["order"] = ["HEAD~2", "HEAD~1", "HEAD"]

        # Add special debug info about the intent
        if intent_name == "DROP" and "target" not in params:
            # Look for last N commits pattern again (failsafe)
            nums = re.findall(r'(\d+)', normalized_text)
            if len(nums) > 0 and "last" in normalized_text and "commit" in normalized_text:
                count = int(nums[0])
                if count > 1:
                    params["target"] = f"HEAD~{count-1}..HEAD"
                else:
                    params["target"] = "HEAD"
        
        # Filter params to only include those in the schema
        if all_params:
            params = {k: v for k, v in params.items() if k in all_params}
            
        return params

    def _is_likely_noise(self, text: str) -> bool:
        """
        Detect if the input is likely noise or random characters.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Boolean indicating if text appears to be noise
        """
        text = text.lower().strip()
        
        # Very short inputs without meaningful content
        if len(text) < 4:
            return True
            
        # Check for at least one domain-relevant word
        words = re.findall(r'\b\w+\b', text)
        has_domain_word = any(word in self.DOMAIN_WORDS for word in words)
        
        # Check for high ratio of non-word characters or random character sequences
        char_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / max(1, len(text))
        high_strange_char_ratio = char_ratio > 0.3
        
        # Check for lack of vowels (common in keyboard mashing)
        consonant_only = True
        for word in words:
            if any(c in 'aeiou' for c in word):
                consonant_only = False
                break
                
        # Check for repeating patterns (like "asdasdasd")
        repeating = False
        if len(text) >= 6:
            for pattern_len in [2, 3]:
                if len(text) >= pattern_len * 3:
                    for i in range(len(text) - pattern_len * 2):
                        pattern = text[i:i+pattern_len]
                        if text[i+pattern_len:i+pattern_len*2] == pattern:
                            repeating = True
                            break
                            
        # Combine factors to determine if it's noise
        return (not has_domain_word or 
                high_strange_char_ratio or 
                consonant_only or 
                repeating)

    def _adjust_confidence_for_noise(self, text: str, confidence: float) -> float:
        """
        Adjust confidence score if the input appears to be noise.
        
        Args:
            text: Input text
            confidence: Original confidence score
            
        Returns:
            Adjusted confidence score
        """
        if self._is_likely_noise(text):
            # Significantly reduce confidence for likely noise
            return min(confidence, self.CONFIDENCE_THRESHOLD - 0.02)
        return confidence

def explain_prediction(classifier: IntentClassifier, text: str) -> Dict[str, Any]:
    """
    Provide an explanation of what features led to the model's prediction.
    Helpful for debugging model behavior.
    
    Args:
        classifier: The classifier instance
        text: The input text
        
    Returns:
        Dictionary with explanation information
    """
    # Apply typo correction
    normalized_text = classifier._normalize_text(text)
    
    # Check if any typos were corrected
    typos_corrected = normalized_text != text.lower().strip()
    
    features = classifier._prepare_features(normalized_text)
    
    try:
        # Get top 3 predictions
        top_predictions = classifier.get_top_n_predictions(text, 3)
        prediction, confidence = top_predictions[0]
        
        # Ensure prediction is a string
        if not isinstance(prediction, str):
            prediction = str(prediction)
        
        # Check if prediction is UNKNOWN
        if prediction.upper() == "UNKNOWN":
            is_noise = classifier._is_likely_noise(text)
            return {
                'prediction': "UNKNOWN",
                'confidence': confidence,
                'alternative_predictions': top_predictions[1:],
                'features': features.iloc[0].to_dict(),
                'is_noise': is_noise,
                'normalized_text': normalized_text,
                'typos_corrected': typos_corrected,
                'noise_factors': {
                    'too_short': len(text.strip()) < 4,
                    'no_domain_words': not any(word in classifier.DOMAIN_WORDS 
                                              for word in re.findall(r'\b\w+\b', text.lower())),
                    'high_special_chars': sum(not c.isalnum() and not c.isspace() for c in text) / max(1, len(text)) > 0.3,
                    'no_vowels': all(c not in 'aeiou' for c in text.lower() if c.isalpha()),
                }
            }
        
        try:
            intent = Intent[prediction.upper()]
        except KeyError:
            intent = None
        
        # Get feature importance if possible
        feature_importances = {}
        if hasattr(classifier.model, 'named_steps') and 'classifier' in classifier.model.named_steps:
            clf = classifier.model.named_steps['classifier']
            if hasattr(clf, 'feature_importances_'):
                for i, importance in enumerate(clf.feature_importances_):
                    if i < len(features.columns):
                        feature_importances[features.columns[i]] = importance
        
        # Extract potential parameters
        params = classifier.extract_intent_parameters(text)
        
        return {
            'prediction': prediction,
            'intent': intent.value if intent else None,
            'confidence': confidence,
            'is_noise': classifier._is_likely_noise(text),
            'normalized_text': normalized_text,
            'typos_corrected': typos_corrected,
            'alternative_predictions': top_predictions[1:],
            'features': features.iloc[0].to_dict(),
            'extracted_keywords': [
                kw for kw, present in {
                    'squash/combine': features['has_squash_keyword'].iloc[0] == 1,
                    'reword/message': features['has_reword_keyword'].iloc[0] == 1,
                    'branch/name': features['has_branch_keyword'].iloc[0] == 1,
                    'rebase/onto': features['has_rebase_keyword'].iloc[0] == 1,
                    'numbers': features['has_number'].iloc[0] == 1
                }.items() if present
            ],
            'word_count': features['word_count'].iloc[0],
            'feature_importances': feature_importances,
            'extracted_parameters': params
        }
    except Exception as e:
        return {
            'error': str(e),
            'features': features.iloc[0].to_dict() if not features.empty else {}
        }


if __name__ == "__main__":
    """
    Simple command-line interface for testing the classifier.
    """
    # Update the CLI to show when corrections are made
    
    try:
        classifier = IntentClassifier()
        
        print("Git-Surgeon Intent Classifier")
        print("Enter natural language query (or 'exit' to quit)")
        
        while True:
            user_input = input("\nQuery: ").strip()
            if user_input.lower() in ('exit', 'quit', 'q'):
                break
            
            if user_input.lower() == 'debug':
                print("\nEnter text to debug classification:")
                debug_text = input("> ")
                explanation = explain_prediction(classifier, debug_text)
                print("\nDebug Information:")
                for key, value in explanation.items():
                    print(f"  {key}: {value}")
                continue
                
            if user_input.lower() == 'cache':
                stats = classifier.cache_stats()
                print("\nCache Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
                
            if user_input.lower() == 'clearcache':
                classifier.clear_cache()
                print("Cache cleared.")
                continue
                
            try:
                # Normalize text for correction
                normalized = classifier._normalize_text(user_input)
                
                # Check if any corrections were made
                if normalized != user_input.lower():
                    # Find what was corrected
                    orig_words = re.findall(r'\b\w+\b', user_input.lower())
                    norm_words = re.findall(r'\b\w+\b', normalized)
                    
                    corrections = []
                    for i, word in enumerate(orig_words):
                        if i < len(norm_words) and norm_words[i] != word:
                            corrections.append(f"'{word}' → '{norm_words[i]}'")
                    
                    if corrections:
                        print(f"Corrected: {', '.join(corrections)}")
                
                # Before making ML predictions, check for special cases
                # Special case for commit hash patterns in REORDER
                hash_pattern = re.findall(r'\b([a-f0-9]{4,40})\b', user_input.lower())
                reorder_words = ['switch', 'swap', 'reorder', 'exchange', 'position']
                
                if len(hash_pattern) >= 2 and any(word in user_input.lower() for word in reorder_words):
                    # Override predictions, this is definitely a REORDER intent
                    intent = "REORDER"
                    intent_enum = Intent.REORDER
                    confidence = 0.95
                    
                    # Extract parameters directly
                    params = {
                        "order": [hash_pattern[1], hash_pattern[0]]  # Swap the first two hashes found
                    }
                    
                    print(f"Intent: {intent_enum.name} ({confidence:.1%} confident)")
                    print("Extracted parameters:")
                    for param, value in params.items():
                        print(f"  - {param}: {value}")
                    
                    # Skip the rest of the loop for this special case
                    continue
                    
                # Standard ML predictions continue here
                # Get top 3 predictions with confidence
                top_predictions = classifier.get_top_n_predictions(user_input, 3)
                intent, confidence = top_predictions[0]
                
                # Ensure intent is a string
                if not isinstance(intent, str):
                    intent = str(intent)
                
                if intent.upper() == "UNKNOWN":
                    print(f"Intent: UNKNOWN (input not recognized as a valid git command)")
                else:
                    intent_enum = Intent[intent.upper()]
                    
                    # Print top prediction with confidence
                    if confidence:
                        print(f"Intent: {intent_enum.name} ({confidence:.1%} confident)")
                    else:
                        print(f"Intent: {intent_enum.name}")
                
                # Print alternatives if available
                if len(top_predictions) > 1:
                    print("Alternative interpretations:")
                    for alt_intent, alt_conf in top_predictions[1:]:
                        try:
                            if alt_intent.upper() == "UNKNOWN":
                                print(f"  - UNKNOWN ({alt_conf:.1%} confident)")
                            else:
                                alt_enum = Intent[alt_intent.upper()]
                                print(f"  - {alt_enum.name} ({alt_conf:.1%} confident)")
                        except (KeyError, AttributeError):
                            pass
                
                # Extract and print parameters if available
                if intent.upper() != "UNKNOWN":
                    params = classifier.extract_intent_parameters(user_input)
                    
                    # Ensure we don't have incorrect parameters
                    # Filter parameters based on the intent schema
                    # ...existing code...
                    
                    # Special handling for REWORD to ensure 'target' parameter is present
                    if intent.upper() == "REWORD" and "target" not in params:
                        params["target"] = "HEAD"
                        
                    # Special handling for REORDER with "reverse" order
                    # If the order parameter is "reverse", it's a valid value and not missing
                    if intent.upper() == "REORDER" and "order" in params and params["order"] == "reverse":
                        # This is a special case where the order is a string value, not an array
                        # The parameter is present and valid, no need to show it as missing
                        pass
                    
                    # Double-check DROP intent with special patterns (extra failsafe)
                    if intent.upper() == "DROP" and "target" not in params and "message_match" not in params:
                        # Get schema
                        schema = INTENT_SCHEMAS.get(intent_enum, {})
                        required_one_of = schema.get("required_one_of", [])
                        
                        # Latest commit cases
                        if "latest" in normalized or "recent" in normalized:
                            params["target"] = "HEAD"
                            
                        # Message-based targets - now use message_match parameter
                        msg_match = re.search(r'[\'"](.*?)[\'"]', normalized)
                        if msg_match and "message_match" not in params:
                            params["message_match"] = msg_match.group(1)
                            # Use a larger history window for message matching
                            if "target" not in params:
                                params["target"] = "HEAD~50..HEAD"
                        
                        # Hash-based targets
                        hash_match = re.search(r'\b([a-f0-9]{6,40})\b', normalized)
                        if hash_match:
                            params["target"] = hash_match.group(1)
                    
                    if params:
                        print("Extracted parameters:")
                        for param, value in params.items():
                            print(f"  - {param}: {value}")
                    
                    # Show missing required parameters based on schema
                    try:
                        schema = INTENT_SCHEMAS.get(intent_enum, {})
                        
                        # Handle both old and new schema formats
                        if isinstance(schema.get("params"), dict):
                            # New format with dictionary of params
                            required_params = [param for param, metadata in schema.get("params", {}).items() 
                                             if metadata.get("required", True)]
                            
                            # Check for required_one_of constraint
                            required_one_of = schema.get("required_one_of", [])
                            if required_one_of:
                                # Only check for missing if none of the required_one_of params are present
                                if not any(param in params for param in required_one_of):
                                    print("Missing parameters - requires at least one of these:")
                                    for param in required_one_of:
                                        if param not in params:
                                            print(f"  - {param}")
                            
                            # Don't show individual missing parameters that are part of required_one_of
                            # if at least one of the required_one_of params is present
                            if required_one_of and any(param in params for param in required_one_of):
                                # Filter out required_one_of params from the missing check
                                required_params = [p for p in required_params if p not in required_one_of]
                        else:
                            # Old format with list of params
                            required_params = schema.get("params", [])
                        
                        # Show any absolutely required params that are missing
                        missing_params = [param for param in required_params if param not in params]
                        
                        # Special case for REORDER intent - "order" might be string "reverse" which is valid
                        if intent.upper() == "REORDER" and "order" in missing_params and "order" in params:
                            missing_params.remove("order")
                        
                        if missing_params:
                            print("Missing required parameters:")
                            for param in missing_params:
                                print(f"  - {param}")
                                
                    except Exception as e:
                        # Don't crash on schema parsing errors
                        print(f"Error checking parameter requirements: {e}")
                    
            except Exception as e:
                print(f"Error classifying intent: {e}")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to train the model first with: python train.py")