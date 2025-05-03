"""
Defense Engine for RobustNLP
Implements various defense mechanisms against adversarial text attacks
"""

import re
import string
import numpy as np
import random
from typing import List, Dict, Tuple, Callable, Union, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import words, wordnet
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from collections import Counter
import spacy
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # For environments where spaCy models aren't pre-installed
    import subprocess
    subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


class DefenseEngine:
    """Main class for defending against adversarial text attacks"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = None):
        """
        Initialize defense engine with specified model
        
        Args:
            model_name: HuggingFace model name for text processing
            device: Device to run models on ('cpu', 'cuda', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize word dictionary
        self.word_set = set(words.words())
        
        # Initialize defense techniques registry
        self.defense_techniques = {
            "spelling_correction": self.spelling_correction,
            "input_sanitization": self.input_sanitization,
            "adversarial_training": self.adversarial_training,
            "randomized_smoothing": self.randomized_smoothing,
            "ensemble_prediction": self.ensemble_prediction,
            "detection": self.detect_adversarial_input,
            "character_normalization": self.character_normalization,
        }
        
        # Anomaly detection model (lazy-loaded)
        self._anomaly_detector = None
        self._vectorizer = None
    def defend_text(self, text: str, task_type: str, threat_info: Dict, defense_level: str = "standard", **kwargs) -> Dict:
        """
        Defend the given text from identified threats.

        Args:
            text (str): The input text to defend.
            task_type (str): The type of task (e.g., "toxicity_detection").
            threat_info (Dict): Information about the identified threats.
            defense_level (str): The level of defense to apply ("minimal", "standard", "aggressive").

        Returns:
            Dict: A dictionary containing the defended text and additional information.
        """
        # Dummy implementation to sanitize the word "attack"
        defended_text = text.replace("attack", "[sanitized]")
        return {
            "defended_text": defended_text
        }
    def defend(self, text: str, defense_type: str, model=None, **kwargs) -> Union[str, Tuple[str, Any]]:
        """
        Apply a defense mechanism to potentially adversarial text
        
        Args:
            text: Text to defend against
            defense_type: Type of defense to use
            model: Model to defend (if needed by the defense)
            **kwargs: Additional arguments for the specific defense
            
        Returns:
            Processed text or tuple of (processed_text, additional_info)
        """
        if defense_type not in self.defense_techniques:
            raise ValueError(f"Unknown defense type: {defense_type}. Available defenses: {list(self.defense_techniques.keys())}")
        
        return self.defense_techniques[defense_type](text, model, **kwargs)
    
    def spelling_correction(self, text: str, model=None, 
                           confidence_threshold: float = 0.8, 
                           max_edit_distance: int = 2, 
                           **kwargs) -> str:
        """
        Correct spelling errors in the text
        
        Args:
            text: Text to correct
            model: Not used for this defense
            confidence_threshold: Threshold for correction confidence
            max_edit_distance: Maximum edit distance for corrections
            
        Returns:
            Corrected text
        """
        words_list = word_tokenize(text)
        corrected_words = []
        
        for word in words_list:
            # Skip punctuation, numbers, very short words
            if not word.isalpha() or len(word) <= 2:
                corrected_words.append(word)
                continue
                
            # Skip words that are already in the dictionary
            if word.lower() in self.word_set:
                corrected_words.append(word)
                continue
                
            # Find potential corrections
            candidates = []
            for dict_word in self.word_set:
                # Quick filter: length difference
                if abs(len(dict_word) - len(word)) > max_edit_distance:
                    continue
                    
                # Only consider words with small edit distance
                distance = edit_distance(word.lower(), dict_word.lower())
                if distance <= max_edit_distance:
                    # Simple confidence score based on edit distance
                    confidence = 1.0 - (distance / max(len(word), len(dict_word)))
                    candidates.append((dict_word, confidence))
            
            # Sort candidates by confidence
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Apply correction if confident enough
            if candidates and candidates[0][1] >= confidence_threshold:
                correction = candidates[0][0]
                # Preserve capitalization
                if word[0].isupper():
                    correction = correction.capitalize()
                corrected_words.append(correction)
            else:
                corrected_words.append(word)  # Keep original if not confident
        
        return ' '.join(corrected_words)
    
    def input_sanitization(self, text: str, model=None, 
                          strip_html: bool = True, 
                          normalize_whitespace: bool = True,
                          filter_profanity: bool = False,
                          normalize_characters: bool = True,
                          **kwargs) -> str:
        """
        Sanitize text input by removing or normalizing potentially adversarial content
        
        Args:
            text: Text to sanitize
            model: Not used for this defense
            strip_html: Whether to remove HTML tags
            normalize_whitespace: Whether to normalize whitespace
            filter_profanity: Whether to filter profanity
            normalize_characters: Whether to normalize Unicode characters
            
        Returns:
            Sanitized text
        """
        # Strip HTML
        if strip_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        if normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Character normalization
        if normalize_characters:
            # Map visually similar characters to ASCII equivalents
            char_map = {
                '𝔞': 'a', '𝐚': 'a', '𝒂': 'a', '𝚊': 'a', '𝗮': 'a', '𝐚': 'a', '𝒂': 'a', '𝖺': 'a', '𝙖': 'a',
                '𝔟': 'b', '𝐛': 'b', '𝒃': 'b', '𝚋': 'b', '𝗯': 'b', '𝐛': 'b', '𝒃': 'b', '𝖻': 'b', '𝙗': 'b',
                '𝔠': 'c', '𝐜': 'c', '𝒄': 'c', '𝚌': 'c', '𝗰': 'c', '𝐜': 'c', '𝒄': 'c', '𝖼': 'c', '𝙘': 'c',
                # ... and so on for all letters
                # Leet speak normalizations
                '4': 'a', '@': 'a', '8': 'b', '(': 'c', '0': 'o', '1': 'i', '!': 'i', '3': 'e',
                ':': 's', '5': 's', '+': 't', '7': 't', '9': 'g', '6': 'g',
                # Unicode homoglyphs
                'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a', 'å': 'a',
                'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
                'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
                'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
                'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
                'ç': 'c', 'ñ': 'n',
                # Symbols used in place of letters
                '¢': 'c', '€': 'e', '£': 'l', '¥': 'y',
            }
            
            # Apply character mapping
            normalized_text = ""
            for char in text:
                normalized_text += char_map.get(char, char)
            
            text = normalized_text
        
        # Filter profanity (simplified - would use a comprehensive list in production)
        if filter_profanity:
            profanity_list = ["fuck", "shit", "ass", "bitch", "damn", "cunt", "dick"]
            for word in profanity_list:
                text = re.sub(r'\b' + word + r'\b', '[filtered]', text, flags=re.IGNORECASE)
        
        return text
    
    def adversarial_training(self, text: str, model, 
                            train_mode: bool = False,
                            attack_types: List[str] = None,
                            attack_params: Dict = None,
                            **kwargs) -> Union[Tuple[torch.Tensor, bool], torch.Tensor]:
        """
        Apply adversarial training or prediction with a model
        
        Args:
            text: Input text
            model: PyTorch model to train or use for prediction
            train_mode: Whether to train the model or just predict
            attack_types: Types of attacks to use in training
            attack_params: Parameters for the attacks
            
        Returns:
            If train_mode: (loss, True)
            Otherwise: model output
        """
        if train_mode:
            if not hasattr(model, 'train') or not hasattr(model, 'forward'):
                raise ValueError("Model must be a PyTorch model with train() and forward() methods")
                
            from threat_engine import ThreatEngine
            
            # Default attack types and parameters
            if attack_types is None:
                attack_types = ["typo", "synonym", "leetspeak"]
            if attack_params is None:
                attack_params = {"severity": 0.2, "target_words": None}
            
            # Create threat engine
            threat_engine = ThreatEngine()
            
            # Generate adversarial examples
            adversarial_texts = []
            for attack_type in attack_types:
                adv_text = threat_engine.generate_attack(text, attack_type, **attack_params)
                adversarial_texts.append(adv_text)
            
            # Add original text
            all_texts = [text] + adversarial_texts
            
            # Train on all examples
            model.train()
            total_loss = 0
            
            for example in all_texts:
                # Tokenize
                inputs = self.tokenizer(example, return_tensors="pt", padding=True, truncation=True).to(self.device)
                
                # Forward pass
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                total_loss += loss.item()
            
            return (total_loss / len(all_texts), True)
        else:
            # Prediction mode
            model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = model(**inputs)
                return outputs.logits
    
    def randomized_smoothing(self, text: str, model, 
                           num_samples: int = 10,
                           augment_fn: Callable = None,
                           **kwargs) -> Tuple[torch.Tensor, float]:
        """
        Apply randomized smoothing to make model predictions more robust
        
        Args:
            text: Input text
            model: Model to use for prediction
            num_samples: Number of random samples to generate
            augment_fn: Function to generate augmented samples
            
        Returns:
            Tuple of (smoothed_prediction, certification_radius)
        """
        model.eval()
        
        # Default augmentation function (simple word dropout)
        if augment_fn is None:
            def default_augment(text, dropout_prob=0.1):
                words = word_tokenize(text)
                kept_words = [w for w in words if random.random() > dropout_prob]
                if not kept_words:  # Ensure at least one word remains
                    kept_words = [random.choice(words)]
                return ' '.join(kept_words)
            
            augment_fn = default_augment
        
        # Generate augmented samples
        augmented_texts = [augment_fn(text) for _ in range(num_samples)]
        
        # Get predictions for all samples
        all_logits = []
        with torch.no_grad():
            for aug_text in augmented_texts:
                inputs = self.tokenizer(aug_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = model(**inputs)
                all_logits.append(outputs.logits)
        
        # Stack logits and compute mean
        if all_logits:
            stacked_logits = torch.cat(all_logits, dim=0)
            mean_logits = torch.mean(stacked_logits, dim=0, keepdim=True)
            
            # Simple certification radius based on prediction confidence
            probs = F.softmax(mean_logits, dim=1)
            top_prob, top_class = torch.max(probs, dim=1)
            certification_radius = top_prob.item() - (1 - top_prob.item()) / (torch.numel(probs) - 1)
            
            return mean_logits, max(0, certification_radius)
        else:
            # Fallback if no logits were generated
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = model(**inputs)
            return outputs.logits, 0.0
    
    def ensemble_prediction(self, text: str, models: List[Any], 
                          weights: List[float] = None,
                          **kwargs) -> torch.Tensor:
        """
        Use an ensemble of models to make a more robust prediction
        
        Args:
            text: Input text
            models: List of models to use
            weights: Optional weights for each model
            
        Returns:
            Ensemble prediction logits
        """
        if not models:
            raise ValueError("At least one model must be provided")
            
        # Equal weights by default
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
            
        # Get predictions from all models
        all_logits = []
        for model in models:
            model.eval()
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = model(**inputs)
                # Ensure all models output logits of the same shape
                all_logits.append(outputs.logits)
        
        # Weight and sum logits
        weighted_logits = [w * logits for w, logits in zip(weights, all_logits)]
        ensemble_logits = sum(weighted_logits)
        
        return ensemble_logits
    
    def detect_adversarial_input(self, text: str, model=None, 
                               threshold: float = 0.8,
                               **kwargs) -> Tuple[str, float]:
        """
        Detect if the input text is likely adversarial
        
        Args:
            text: Input text to check
            model: Not used for this defense
            threshold: Threshold for classification as adversarial
            
        Returns:
            Tuple of (original_text, adversarial_score)
        """
        # Lazy-load the anomaly detector
        if self._anomaly_detector is None:
            self._train_anomaly_detector()
        
        # Extract features
        features = self._extract_anomaly_features(text)
        
        # Get abnormality score from the isolation forest
        anomaly_score = self._anomaly_detector.score_samples([features])[0]
        # Convert to probability-like score (0-1 range, higher means more adversarial)
        adversarial_score = 1 - (anomaly_score + 0.5)  # Typical range is (-0.5, 0.5)
        
        return text, min(max(adversarial_score, 0.0), 1.0)
    
    def _train_anomaly_detector(self, sample_texts: List[str] = None):
        """Train an anomaly detector for adversarial text detection"""
        if sample_texts is None:
            # Use some default texts if none provided (in real use case, would use a large corpus)
            sample_texts = [
                "This is a normal text example for training.",
                "Another standard text without any adversarial content.",
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning models can be vulnerable to adversarial examples.",
                "Natural language processing involves understanding human language.",
                "Artificial intelligence systems are designed to mimic human cognition.",
                "The weather today is sunny with a chance of rain later."
            ]
        
        # Extract features from the sample texts
        features_list = [self._extract_anomaly_features(text) for text in sample_texts]
        
        # Train an isolation forest model
        self._anomaly_detector = IsolationForest(random_state=42, contamination=0.1)
        self._anomaly_detector.fit(features_list)
    
    def _extract_anomaly_features(self, text: str) -> np.ndarray:
        """Extract features for anomaly detection"""
        # Calculate various statistical features that might indicate adversarial content
        
        # Character-level features
        char_counts = Counter(text.lower())
        alpha_ratio = sum(char_counts[c] for c in string.ascii_lowercase) / max(1, len(text))
        digit_ratio = sum(char_counts[c] for c in string.digits) / max(1, len(text))
        special_char_ratio = 1 - alpha_ratio - digit_ratio
        
        # Word-level features
        words = word_tokenize(text.lower())
        avg_word_len = sum(len(w) for w in words) / max(1, len(words))
        
        # Dictionary word ratio
        dict_word_ratio = sum(1 for w in words if w in self.word_set) / max(1, len(words))
        
        # Repeated character ratio
        repeated_chars = sum(1 for i in range(1, len(text)) if text[i] == text[i-1])
        repeated_char_ratio = repeated_chars / max(1, len(text) - 1)
        
        # N-gram character entropy (simplified)
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        bigram_counts = Counter(bigrams)
        bigram_probs = [count / max(1, len(bigrams)) for count in bigram_counts.values()]
        char_entropy = -sum(p * np.log2(p) for p in bigram_probs) if bigram_probs else 0
        
        # Unicode range usage
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(1, len(text))
        
        # Combine features
        features = np.array([
            alpha_ratio, 
            digit_ratio, 
            special_char_ratio,
            avg_word_len, 
            dict_word_ratio,
            repeated_char_ratio,
            char_entropy,
            ascii_ratio
        ])
        
        return features
    
    def character_normalization(self, text: str, model=None, **kwargs) -> str:
        """
        Normalize unicode characters and potential visual attack patterns
        
        Args:
            text: Text to normalize
            model: Not used for this defense
            
        Returns:
            Normalized text
        """
        # Create a map of visually similar characters to their ASCII equivalents
        char_map = {
            # Homoglyphs
            'а': 'a',  # Cyrillic 'a' to Latin 'a'
            'е': 'e',  # Cyrillic 'e' to Latin 'e'
            'о': 'o',  # Cyrillic 'o' to Latin 'o'
            'р': 'p',  # Cyrillic 'p' to Latin 'p'
            'с': 'c',  # Cyrillic 'c' to Latin 'c'
            'ѕ': 's',  # Cyrillic 's' to Latin 's'
            
            # Leet speak
            '4': 'a',
            '@': 'a',
            '8': 'b',
            '(': 'c',
            '0': 'o',
            '1': 'i',
            '!': 'i',
            '3': 'e',
            ':': 's',
            '5': 's',
            '+': 't',
            '7': 't',
            
            # Accented characters to ASCII
            'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a', 'ã': 'a', 'å': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o', 'õ': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n',
            'ÿ': 'y',
            
            # Other Unicode characters sometimes used in adversarial content
            '𝒂': 'a', '𝒃': 'b', '𝒄': 'c', '𝒅': 'd', '𝒆': 'e', '𝒇': 'f', 
            '𝒈': 'g', '𝒉': 'h', '𝒊': 'i', '𝒋': 'j', '𝒌': 'k', '𝒍': 'l', 
            '𝒎': 'm', '𝒏': 'n', '𝒐': 'o', '𝒑': 'p', '𝒒': 'q', '𝒓': 'r', 
            '𝒔': 's', '𝒕': 't', '𝒖': 'u', '𝒗': 'v', '𝒘': 'w', '𝒙': 'x', 
            '𝒚': 'y', '𝒛': 'z',
        }
        
        # Normalize text
        normalized_text = ""
        for char in text:
            normalized_text += char_map.get(char, char)
            
        return normalized_text
    
    def defense_pipeline(self, text: str, defense_sequence: List[Dict], model=None, **kwargs) -> str:
        """
        Apply a sequence of defenses in pipeline fashion
        
        Args:
            text: Original text to defend
            defense_sequence: List of dictionaries with defense configurations
                Example: [
                    {"type": "character_normalization"},
                    {"type": "spelling_correction", "confidence_threshold": 0.7}
                ]
            model: Model to use for defenses that require it
            
        Returns:
            Text after applying all defenses in sequence
        """
        result = text
        for defense_config in defense_sequence:
            defense_type = defense_config.pop("type")
            defense_result = self.defend(result, defense_type, model, **defense_config)
            
            # Handle defenses that return tuples
            if isinstance(defense_result, tuple):
                result = defense_result[0]
            else:
                result = defense_result
            
            # Restore the config for potential reuse
            defense_config["type"] = defense_type
        
        return result


# Example usage
if __name__ == "__main__":
    defense_engine = DefenseEngine()
    
    # Example adversarial text
    adversarial_text = "Th1s is a adv3rs4rial text w1th s0me sp3ll1ng err0rs and ch4racter repl4cem3nts."
    
    # Apply different defenses
    corrected_text = defense_engine.defend(adversarial_text, "spelling_correction")
    print(f"Spelling corrected: {corrected_text}")
    
    sanitized_text = defense_engine.defend(adversarial_text, "input_sanitization")
    print(f"Sanitized: {sanitized_text}")
    
    normalized_text = defense_engine.defend(adversarial_text, "character_normalization")
    print(f"Normalized: {normalized_text}")
    
    # Check if text is adversarial
    _, adv_score = defense_engine.defend(adversarial_text, "detection")
    print(f"Adversarial score: {adv_score}")
    
    # Apply defense pipeline
    pipeline_result = defense_engine.defense_pipeline(
        adversarial_text,
        [
            {"type": "character_normalization"},
            {"type": "spelling_correction", "confidence_threshold": 0.7}
        ]
    )
    print(f"Pipeline result: {pipeline_result}")