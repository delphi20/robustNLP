from textattack.attack_recipes import TextFoolerJin2019, BERTAttackLi2020
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_results import SuccessfulAttackResult
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import nltk 


class ThreatGenerator:
    """
    Generates adversarial examples using TextFooler and BERT-Attack methods.
    """
    
    def __init__(self, model_name='bert-base-uncased', attack_method='textfooler', device=None):
        self.model_name = model_name
        self.attack_method = attack_method
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create TextAttack model wrapper
        self.model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer)
        
        # Initialize attack methods
        self.attacks = {}
        if attack_method in ['textfooler', 'all']:
            self.attacks['textfooler'] = TextFoolerJin2019.build(self.model_wrapper)
        if attack_method in ['bert-attack', 'all']:
            self.attacks['bert-attack'] = BERTAttackLi2020.build(self.model_wrapper)
    
    def generate_adversarial_examples(self, texts, labels, attack_method=None):
        """
        Generate adversarial examples for the given texts.
        
        Args:
            texts (list): List of input texts
            labels (list): List of corresponding labels
            attack_method (str, optional): Override the attack method
            
        Returns:
            dict: Dictionary containing original texts, adversarial texts, and success flags
        """
        if attack_method is None:
            attack_method = self.attack_method
            
        if attack_method == 'all':
            attack_method = 'textfooler'  # Default to TextFooler if 'all' is specified
            
        attack = self.attacks.get(attack_method)
        if attack is None:
            raise ValueError(f"Attack method {attack_method} not initialized")
        
        results = {
            'original_text': [],
            'adversarial_text': [],
            'original_label': [],
            'adversarial_label': [],
            'success': []
        }
        
        for text, label in zip(texts, labels):
            # Create a dataset for TextAttack
            attack_result = attack.attack(text, label)
            
            results['original_text'].append(text)
            results['original_label'].append(label)
            
            if isinstance(attack_result, SuccessfulAttackResult):
                results['adversarial_text'].append(attack_result.perturbed_text())
                results['adversarial_label'].append(attack_result.output)
                results['success'].append(True)
            else:
                results['adversarial_text'].append(text)  # No change if attack failed
                results['adversarial_label'].append(label)
                results['success'].append(False)
                
        return results
    
    def get_available_attack_methods(self):
        """Return a list of available attack methods."""
        return list(self.attacks.keys())

