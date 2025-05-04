"""
Threat Generation Engine for RobustNLP
Implements various adversarial attack techniques for text
"""
import random
import string
from typing import List, Dict, Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


class ThreatEngine:
    """Main class for generating adversarial text attacks"""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.attack_techniques = {
            "typo": self.typo_attack,
            "synonym": self.synonym_attack,
            "leetspeak": self.leetspeak_attack,
            "bert_attack": self.bert_attack
        }

    def analyze_text(self, text: str, task_type: str = "toxicity_detection", attack_type: str = None, severity: float = 0.3) -> Dict:
        """
        Analyze or modify text using the specified attack.

        Args:
            text (str): The input text to analyze.
            task_type (str): The type of analysis task (e.g., "toxicity_detection").
            attack_type (str): The type of attack (e.g., "typo", "synonym"). Defaults to None.
            severity (float): The severity of the attack.

        Returns:
            Dict: A dictionary with analysis or modification results.
        """
        try:
            if attack_type:
                # Apply adversarial attack
                modified_text = self.generate_attack(text, attack_type, severity)
                return {
                    "original_text": text,
                    "modified_text": modified_text,
                    "attack_type": attack_type,
                    "severity": severity,
                }
            else:
                # Analyze text for threats or toxicity
                if "attack" in text.lower():
                    return {
                        "toxicity_score": 0.8,
                        "threat_detected": True,
                        "threat_type": "harmful_content",
                        "threat_score": 0.9
                    }
                else:
                    return {
                        "toxicity_score": 0.2,
                        "threat_detected": False
                    }
        except Exception as e:
            return {"error": str(e)}
    def typo_attack(self, text: str, severity: float = 0.3, target_words: List[str] = None, **kwargs) -> str:
        words, indices = self.preprocess_text(text, severity, target_words)
        for idx in indices:
            word = words[idx]
            typo_type = random.choice(["swap", "delete", "insert", "replace"])
            if typo_type == "swap" and len(word) > 1:
                pos = random.randint(0, len(word) - 2)
                word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            elif typo_type == "delete" and len(word) > 1:
                pos = random.randint(0, len(word) - 1)
                word = word[:pos] + word[pos+1:]
            elif typo_type == "insert":
                pos = random.randint(0, len(word))
                char = random.choice(string.ascii_lowercase)
                word = word[:pos] + char + word[pos:]
            elif typo_type == "replace" and len(word) > 0:
                pos = random.randint(0, len(word) - 1)
                word = word[:pos] + random.choice(string.ascii_lowercase) + word[pos+1:]
            words[idx] = word
        return ' '.join(words)

    def preprocess_text(self, text: str, severity: float, target_words: List[str] = None) -> Tuple[List[str], List[int]]:
        words = word_tokenize(text)
        num_to_modify = max(1, int(len(words) * severity))
        if target_words:
            indices = [i for i, word in enumerate(words) if word.lower() in [t.lower() for t in target_words]]
        else:
            indices = list(range(len(words)))
        indices = [i for i in indices if len(words[i]) > 2 and words[i].isalpha()]
        return words, indices[:num_to_modify]
        
    def synonym_attack(self, text: str, severity: float = 0.3, target_words: List[str] = None, 
                        pos_filter: List[str] = None, **kwargs) -> str:
        """
        Replace words with their synonyms
        
        Args:
            text: Original text to attack
            severity: Percentage of words to modify
            target_words: Specific words to target for replacement
            pos_filter: Only target words with these parts of speech ('n', 'v', 'a', 'r')
            
        Returns:
            Text with synonym replacements
        """
        words = word_tokenize(text)
        num_to_modify = max(1, int(len(words) * severity))
        
        # Filter to target words if specified
        if target_words:
            indices = [i for i, word in enumerate(words) if word.lower() in [t.lower() for t in target_words]]
        else:
            indices = list(range(len(words)))
            
        # Skip very short words and non-alphabetic tokens
        indices = [i for i in indices if len(words[i]) > 2 and words[i].isalpha()]
        
        if not indices:
            return text
            
        # Select random words to modify
        indices_to_modify = random.sample(indices, min(num_to_modify, len(indices)))
        
        for idx in indices_to_modify:
            word = words[idx]
            synonyms = []
            
            # Find all possible synonyms
            for syn in wordnet.synsets(word):
                # Filter by part of speech if specified
                if pos_filter and syn.pos() not in pos_filter:
                    continue
                    
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower() and synonym.isalpha():
                        synonyms.append(synonym)
            
            if synonyms:
                replacement = random.choice(synonyms)
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[idx] = replacement
        
        return ' '.join(words)
    
    def leetspeak_attack(self, text: str, severity: float = 0.3, target_words: List[str] = None, **kwargs) -> str:
        """
        Convert text to leetspeak (e.g., "elite" -> "3l1t3")
        
        Args:
            text: Original text to attack
            severity: How aggressive the leet conversion should be (0.0-1.0)
            target_words: Specific words to target for leet conversion
            
        Returns:
            Leetspeak text
        """
        leet_map = {
            'a': ['4', '@', 'Д'],
            'b': ['8', '6', 'ß'],
            'c': ['(', '{', '[', '<'],
            'd': [')', 'D', 'Ð'],
            'e': ['3', '€', 'ë'],
            'f': ['ph', 'ƒ'],
            'g': ['6', '9', 'G'],
            'h': ['#', '/-/', '[-]', ']-[', '}-{', '}{'],
            'i': ['1', '!', '|', 'l', 'ï'],
            'j': [';', 'ʝ'],
            'k': ['X', '|<', '|{'],
            'l': ['1', '|', '£', '7'],
            'm': ['|\\/|', '/\\/\\', '//\\\\//'],
            'n': ['|\\|', '/\\/', 'И', 'η'],
            'o': ['0', 'Ø', 'ö'],
            'p': ['|°', '|>', '|*', 'þ'],
            'q': ['O_', '9', '(,)'],
            'r': ['|2', 'Я', '®'],
            's': ['5', '$', 'z'],
            't': ['7', '+', '†'],
            'u': ['|_|', 'µ', 'ü'],
            'v': ['\\/', '√'],
            'w': ['\\/\\/', 'vv', '\\N', 'Ш'],
            'x': ['><', 'Ж', '}{'],
            'y': ['j', '`/', 'Ý', '¥'],
            'z': ['2', '7_', '%']
        }
        
        words = word_tokenize(text)
        num_to_modify = max(1, int(len(words) * severity))
        
        # Filter to target words if specified
        if target_words:
            indices = [i for i, word in enumerate(words) if word.lower() in [t.lower() for t in target_words]]
        else:
            indices = list(range(len(words)))
            
        # Skip very short words and non-alphabetic tokens
        indices = [i for i in indices if len(words[i]) > 2 and words[i].isalpha()]
        
        if not indices:
            return text
            
        # Select random words to modify
        indices_to_modify = random.sample(indices, min(num_to_modify, len(indices)))
        
        for idx in indices_to_modify:
            word = words[idx]
            leet_word = ""
            
            for char in word:
                lower_char = char.lower()
                if lower_char in leet_map and random.random() < severity:
                    leet_char = random.choice(leet_map[lower_char])
                    # Preserve capitalization
                    if char.isupper():
                        leet_char = leet_char.upper() if len(leet_char) == 1 else leet_char
                    leet_word += leet_char
                else:
                    leet_word += char
                    
            words[idx] = leet_word
        
        return ' '.join(words)
    
    def bert_attack(self, text: str, severity: float = 0.3, target_words: List[str] = None, 
                   max_candidates: int = 50, **kwargs) -> str:
        """
        Use BERT to generate contextual word replacements
        
        Args:
            text: Original text to attack
            severity: Percentage of words to modify
            target_words: Specific words to target for replacement
            max_candidates: Maximum number of candidate replacements to consider
            
        Returns:
            Text with BERT-suggested replacements
        """
        words = word_tokenize(text)
        num_to_modify = max(1, int(len(words) * severity))
        
        # Filter to target words if specified
        if target_words:
            indices = [i for i, word in enumerate(words) if word.lower() in [t.lower() for t in target_words]]
        else:
            indices = list(range(len(words)))
            
        # Skip very short words and non-alphabetic tokens
        indices = [i for i in indices if len(words[i]) > 2 and words[i].isalpha()]
        
        if not indices:
            return text
            
        # Select random words to modify
        indices_to_modify = random.sample(indices, min(num_to_modify, len(indices)))
        
        for idx in indices_to_modify:
            word = words[idx]
            
            # Create a masked version of the sentence
            masked_words = words.copy()
            masked_words[idx] = self.tokenizer.mask_token
            masked_text = ' '.join(masked_words)
            
            # Get BERT predictions for the masked token
            inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Find the position of the masked token
            mask_idx = inputs.input_ids[0].tolist().index(self.tokenizer.mask_token_id)
            
            # Get the top predictions
            logits = outputs.logits[0, mask_idx, :]
            probs = torch.softmax(logits, dim=0)
            top_k_probs, top_k_indices = torch.topk(probs, max_candidates)
            
            # Filter candidates
            candidates = []
            for i, idx in enumerate(top_k_indices):
                token = self.tokenizer.convert_ids_to_tokens([idx.item()])[0]
                
                # Skip special tokens and subwords starting with ##
                if token in self.tokenizer.all_special_tokens or token.startswith('##'):
                    continue
                    
                # Skip the original word
                if token.lower() == word.lower():
                    continue
                    
                # Check if token is alphabetic after removing ##
                clean_token = token.replace('##', '')
                if not clean_token.isalpha():
                    continue
                    
                candidates.append(token)
                if len(candidates) >= 5:  # Limit to 5 actual candidates
                    break
            
            if candidates:
                replacement = random.choice(candidates)
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[idx] = replacement
        
        return ' '.join(words)
    
    def wordbugs_attack(self, text: str, severity: float = 0.3, target_words: List[str] = None, **kwargs) -> str:
        """
        Implementation of DeepWordBug attack that makes minimal but effective perturbations
        
        Args:
            text: Original text to attack
            severity: Percentage of words to modify
            target_words: Specific words to target for bugs
            
        Returns:
            Text with subtle character-level perturbations
        """
        def _bug_word(word: str) -> str:
            if len(word) <= 2:
                return word
                
            bug_type = random.choice(["swap", "substitute", "delete", "insert", "repeat"])
            
            if bug_type == "swap" and len(word) > 2:
                # Swap internal characters (not first or last)
                pos = random.randint(1, len(word) - 2)
                return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                
            elif bug_type == "substitute" and len(word) > 0:
                # Substitute with visually similar character
                visual_map = {
                    'a': ['@', 'α', '4'],
                    'b': ['6', '8'],
                    'c': ['(', '¢'],
                    'd': ['ð'],
                    'e': ['3', '€'],
                    'g': ['9', '6'],
                    'h': ['#'],
                    'i': ['1', '!', '|'],
                    'l': ['1', '|'],
                    'o': ['0', 'ο'],
                    's': ['$', '5'],
                    't': ['7', '+'],
                    'x': ['×'],
                    'z': ['2']
                }
                
                # Choose a position not at beginning for better readability
                pos = random.randint(1, len(word) - 1)
                char = word[pos].lower()
                
                if char in visual_map:
                    replacement = random.choice(visual_map[char])
                    if word[pos].isupper() and len(replacement) == 1:
                        replacement = replacement.upper()
                    return word[:pos] + replacement + word[pos+1:]
                return word
                
            elif bug_type == "delete" and len(word) > 2:
                # Delete an internal character (not first or last for readability)
                pos = random.randint(1, len(word) - 2)
                return word[:pos] + word[pos+1:]
                
            elif bug_type == "insert":
                # Insert a similar character
                pos = random.randint(1, len(word) - 1)  # Insert internally
                
                # Choose a similar character to the one at this position
                char = word[pos].lower()
                similar_chars = {
                    'a': ['s', 'q', 'z'],
                    'b': ['v', 'n', 'h'],
                    'c': ['x', 'v', 'd'],
                    'd': ['s', 'f', 'e'],
                    'e': ['w', 'r', 'd'],
                    'f': ['d', 'g', 'r'],
                    'g': ['f', 'h', 't'],
                    'h': ['g', 'j', 'y'],
                    'i': ['u', 'o', 'k'],
                    'j': ['h', 'k', 'u'],
                    'k': ['j', 'l', 'i'],
                    'l': ['k', 'o'],
                    'm': ['n', 'j'],
                    'n': ['m', 'b', 'h'],
                    'o': ['i', 'p', 'l'],
                    'p': ['o', 'l'],
                    'q': ['w', 'a'],
                    'r': ['e', 't', 'f'],
                    's': ['a', 'd', 'w'],
                    't': ['r', 'y', 'g'],
                    'u': ['y', 'i', 'j'],
                    'v': ['c', 'b', 'f'],
                    'w': ['q', 'e', 's'],
                    'x': ['z', 'c', 'd'],
                    'y': ['t', 'u', 'h'],
                    'z': ['a', 'x', 's']
                }
                
                if char in similar_chars:
                    insert_char = random.choice(similar_chars[char])
                    if word[pos].isupper():
                        insert_char = insert_char.upper()
                    return word[:pos] + insert_char + word[pos:]
                return word
                
            elif bug_type == "repeat" and len(word) > 0:
                # Repeat a character
                pos = random.randint(0, len(word) - 1)
                return word[:pos+1] + word[pos] + word[pos+1:]
                
            return word  # Default fallback

        words = word_tokenize(text)
        num_to_modify = max(1, int(len(words) * severity))
        
        # Filter to target words if specified
        if target_words:
            indices = [i for i, word in enumerate(words) if word.lower() in [t.lower() for t in target_words]]
        else:
            indices = list(range(len(words)))
            
        # Skip very short words and non-alphabetic tokens
        indices = [i for i in indices if len(words[i]) > 3 and words[i].isalpha()]
        
        if not indices:
            return text
            
        # Select random words to modify
        indices_to_modify = random.sample(indices, min(num_to_modify, len(indices)))
        
        for idx in indices_to_modify:
            words[idx] = _bug_word(words[idx])
        
        return ' '.join(words)
    
    def hotflip_attack(self, text: str, severity: float = 0.3, target_words: List[str] = None, **kwargs) -> str:
        """
        Simplified HotFlip-inspired attack that flips characters in words
        
        Args:
            text: Original text to attack
            severity: Percentage of words to modify
            target_words: Specific words to target for flips
            
        Returns:
            Text with character flips
        """
        def _flip_word(word: str) -> str:
            if len(word) <= 2:
                return word
                
            # Common character flips from HotFlip paper
            flip_map = {
                'a': ['e', 'i', 'o', 's'],
                'b': ['d', 'h', 'p', 'v'],
                'c': ['e', 'k', 'o', 's', 'x'],
                'd': ['b', 'o', 'p', 's'],
                'e': ['a', 'c', 'i', 'o', 'r'],
                'f': ['c', 'p', 't'],
                'g': ['d', 'f', 'h', 't'],
                'h': ['b', 'g', 'k', 'm', 'n'],
                'i': ['a', 'e', 'j', 'l', 'o', 'y'],
                'j': ['i', 'g', 'k', 'l'],
                'k': ['c', 'h', 'j', 'l', 'x'],
                'l': ['i', 'j', 'k', 'r', 't'],
                'm': ['h', 'n', 'w'],
                'n': ['h', 'm', 'r'],
                'o': ['a', 'c', 'e', 'i', 'p', 'u'],
                'p': ['b', 'd', 'f', 'o'],
                'q': ['g', 'p'],
                'r': ['e', 'l', 'n', 't'],
                's': ['a', 'c', 'd', 'z'],
                't': ['f', 'g', 'l', 'r', 'y'],
                'u': ['o', 'v', 'w', 'y'],
                'v': ['b', 'u', 'w'],
                'w': ['m', 'u', 'v'],
                'x': ['c', 'k', 's', 'z'],
                'y': ['i', 't', 'u'],
                'z': ['s', 'x']
            }
            
            # Choose a position to flip (avoid first letter for better readability)
            pos = random.randint(1, len(word) - 1)
            char = word[pos].lower()
            
            if char in flip_map:
                replacement = random.choice(flip_map[char])
                # Preserve capitalization
                if word[pos].isupper():
                    replacement = replacement.upper()
                
                return word[:pos] + replacement + word[pos+1:]
            
            return word

        words = word_tokenize(text)
        num_to_modify = max(1, int(len(words) * severity))
        
        # Filter to target words if specified
        if target_words:
            indices = [i for i, word in enumerate(words) if word.lower() in [t.lower() for t in target_words]]
        else:
            indices = list(range(len(words)))
            
        # Skip very short words and non-alphabetic tokens
        indices = [i for i in indices if len(words[i]) > 3 and words[i].isalpha()]
        
        if not indices:
            return text
            
        # Select random words to modify
        indices_to_modify = random.sample(indices, min(num_to_modify, len(indices)))
        
        for idx in indices_to_modify:
            words[idx] = _flip_word(words[idx])
        
        return ' '.join(words)
    
    def attack_pipeline(self, text: str, attack_sequence: List[Dict], **kwargs) -> str:
        """
        Apply a sequence of attacks in pipeline fashion
        
        Args:
            text: Original text to attack
            attack_sequence: List of dictionaries with attack configurations
                Example: [
                    {"type": "typo", "severity": 0.2},
                    {"type": "synonym", "severity": 0.1, "pos_filter": ["n", "v"]}
                ]
            
        Returns:
            Text after applying all attacks in sequence
        """
        result = text
        for attack_config in attack_sequence:
            attack_type = attack_config.pop("type")
            result = self.generate_attack(result, attack_type, **attack_config)
        
        return result
    
    def generate_contextual_attack(self, text: str, model_predictor: Callable, 
                                 target_label: Union[int, str] = None,
                                 max_iterations: int = 20,
                                 early_stop: bool = True,
                                 **kwargs) -> Tuple[str, bool]:
        """
        Generate an attack optimized to fool a specific model
        
        Args:
            text: Original text to attack
            model_predictor: Function that returns prediction given a text
            target_label: Target label (for targeted attacks) or None (for untargeted)
            max_iterations: Maximum number of attack iterations
            early_stop: Whether to stop once attack succeeds
            
        Returns:
            Tuple of (adversarial_text, success_flag)
        """
        # Get original prediction
        original_pred = model_predictor(text)
        current_text = text
        
        # For untargeted attacks, just aim for any label other than original
        if target_label is None:
            target_label = original_pred
            is_targeted = False
        else:
            is_targeted = True
            
        # Try different attack techniques
        attack_types = ["bert_attack", "synonym", "typo", "wordbugs"]
        severities = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for iteration in range(max_iterations):
            # Try a random attack configuration
            attack_type = random.choice(attack_types)
            severity = random.choice(severities)
            
            # Apply the attack
            adversarial_text = self.generate_attack(current_text, attack_type, severity=severity)
            
            # Check if attack succeeded
            pred = model_predictor(adversarial_text)
            
            if (is_targeted and pred == target_label) or (not is_targeted and pred != target_label):
                return adversarial_text, True
                
            # Update text for next iteration (with a bit of exploration)
            if random.random() < 0.7:  # 70% chance to keep changes
                current_text = adversarial_text
        
        # If we get here, we failed to find a successful attack
        return current_text, False


# Example usage
if __name__ == "__main__":
    attack_engine = ThreatEngine()
    
    # Example text
    text = "This is a very important message about security that should not be bypassed by malicious inputs."
    
    # Try different attacks
    typo_attack = attack_engine.generate_attack(text, "typo", severity=0.4)
    print(f"Typo attack: {typo_attack}")
    
    synonym_attack = attack_engine.generate_attack(text, "synonym", severity=0.3)
    print(f"Synonym attack: {synonym_attack}")
    
    leet_attack = attack_engine.generate_attack(text, "leetspeak", severity=0.5)
    print(f"Leetspeak attack: {leet_attack}")
    
    bert_attack = attack_engine.generate_attack(text, "bert_attack", severity=0.3)
    print(f"BERT attack: {bert_attack}")
    
    wordbugs_attack = attack_engine.generate_attack(text, "wordbugs", severity=0.4)
    print(f"WordBugs attack: {wordbugs_attack}")
    
    hotflip_attack = attack_engine.generate_attack(text, "hotflip", severity=0.3)
    print(f"HotFlip attack: {hotflip_attack}")
    
    # Combined attack pipeline
    pipeline_attack = attack_engine.attack_pipeline(
        text,
        [
            {"type": "synonym", "severity": 0.2},
            {"type": "typo", "severity": 0.1},
            {"type": "leetspeak", "severity": 0.1}
        ]
    )
    print(f"Pipeline attack: {pipeline_attack}")