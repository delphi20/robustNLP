# defense_engine/defender.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import nltk
from nltk.corpus import words as nltk_words
import re
import string
from collections import Counter

class Defender:
    """
    Implements defense mechanisms against adversarial text attacks.
    """
    
    def __init__(self, model_name='bert-base-uncased', defense_method='both', device=None):
        self.model_name = model_name
        self.defense_method = defense_method
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Original (undefended) model
        self.original_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        # Initialize preprocessing tools
        if defense_method in ['preprocessing', 'both']:
            nltk.download('words', quiet=True)
            self.nltk_words_set = set(w.lower() for w in nltk_words.words())
    
    def load_dataset(self, dataset_name, split='train[:1000]'):
        """
        Load a dataset from Hugging Face datasets.
        
        Args:
            dataset_name (str): Name of the dataset
            split (str): Dataset split to use
            
        Returns:
            datasets.Dataset: The loaded dataset
        """
        if dataset_name == 'imdb':
            dataset = load_dataset('imdb', split=split)
            # dataset = dataset.rename_column('text', 'text')
            # dataset = dataset.rename_column('label', 'label')
        elif dataset_name == 'sst2':
            dataset = load_dataset('glue', 'sst2', split=split)
            dataset = dataset.rename_column('sentence', 'text')
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        return dataset
    
    def preprocess_text(self, text):
        """
        Apply preprocessing to defend against adversarial attacks.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if self.defense_method not in ['preprocessing', 'both']:
            return text
        
        # Convert to lowercase
        text = text.lower()
        
        # Correct misspellings using a simple approach
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            if word not in self.nltk_words_set and len(word) > 2:
                # Simple spell correction - find closest word
                candidates = [w for w in self.nltk_words_set if abs(len(w) - len(word)) <= 2]
                
                # Find the most similar word using edit distance
                best_word = None
                min_distance = float('inf')
                
                for candidate in candidates[:100]:  # Limit to 100 candidates for efficiency
                    distance = self._levenshtein_distance(word, candidate)
                    if distance < min_distance:
                        min_distance = distance
                        best_word = candidate
                
                if best_word and min_distance <= 2:
                    text = re.sub(r'\b' + re.escape(word) + r'\b', best_word, text)
        
        return text
    
    def _levenshtein_distance(self, s1, s2):
        """
        Calculate the Levenshtein distance between two strings.
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def train(self, dataset, threat_generator=None):
        """
        Train the model using adversarial training if specified.
        
        Args:
            dataset: The dataset to train on
            threat_generator: ThreatGenerator to generate adversarial examples
        """
        # Prepare dataset for training
        def tokenize_function(examples):
            if self.defense_method in ['preprocessing', 'both']:
                examples['text'] = [self.preprocess_text(text) for text in examples['text']]
            return self.tokenizer(examples['text'], padding='max_length', truncation=True)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Add adversarial examples if using adversarial training
        if self.defense_method in ['adversarial_training', 'both'] and threat_generator is not None:
            # Generate adversarial examples for a subset of the dataset
            subset = dataset.select(range(min(100, len(dataset))))
            adv_results = threat_generator.generate_adversarial_examples(
                subset['text'], subset['label']
            )
            
            # Add successful adversarial examples to the training data
            successful_indices = [i for i, success in enumerate(adv_results['success']) if success]
            
            if successful_indices:
                adv_texts = [adv_results['adversarial_text'][i] for i in successful_indices]
                adv_labels = [adv_results['adversarial_label'][i] for i in successful_indices]
                
                # Add adversarial examples to the dataset
                adv_dataset = dataset.from_dict({
                    'text': adv_texts,
                    'label': adv_labels
                })
                tokenized_adv_dataset = adv_dataset.map(tokenize_function, batched=True)
                tokenized_dataset = tokenized_dataset.concatenate(tokenized_adv_dataset)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        
        # Create trainer and train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        trainer.train()
    
    def predict(self, texts, use_preprocessing=True):
        """
        Make predictions on the given texts.
        
        Args:
            texts (list): List of input texts
            use_preprocessing (bool): Whether to apply preprocessing
            
        Returns:
            dict: Dictionary containing predictions and scores
        """
        processed_texts = [self.preprocess_text(text) if use_preprocessing else text for text in texts]
        
        inputs = self.tokenizer(processed_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()
        scores = torch.softmax(logits, dim=1).tolist()
        
        return {
            'predictions': predictions,
            'scores': scores
        }
    
    def predict_original_model(self, texts):
        """
        Make predictions using the original model.
        
        Args:
            texts (list): List of input texts
            
        Returns:
            dict: Dictionary containing predictions and scores
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.original_model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()
        scores = torch.softmax(logits, dim=1).tolist()
        
        return {
            'predictions': predictions,
            'scores': scores
        }
    
    def save_model(self, output_dir):
        """
        Save the defended model.
        
        Args:
            output_dir (str): Directory to save the model
        """
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)