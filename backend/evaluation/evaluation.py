# evaluation/evaluator.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class Evaluator:
    """
    Evaluates the performance of the defense system and visualizes results.
    """
    
    def __init__(self, device=None):
        self.device = device
    
    def evaluate(self, dataset, threat_generator, defender, num_samples=100):
        """
        Evaluate the defense system against adversarial attacks.
        
        Args:
            dataset: The dataset to evaluate on
            threat_generator: ThreatGenerator instance
            defender: Defender instance
            num_samples (int): Number of samples to evaluate
            
        Returns:
            dict: Evaluation results including metrics and examples
        """
        # Limit to a subset for evaluation
        subset = dataset.select(range(min(num_samples, len(dataset))))
        
        # Generate adversarial examples
        adv_results = {}
        for attack_method in threat_generator.get_available_attack_methods():
            adv_results[attack_method] = threat_generator.generate_adversarial_examples(
                subset['text'], subset['label'], attack_method=attack_method
            )
        
        # Evaluate original and defended models
        results = {
            'metrics': {},
            'examples': []
        }
        
        for attack_method, attack_results in adv_results.items():
            # Evaluate original model on adversarial examples
            original_preds = defender.predict_original_model(attack_results['adversarial_text'])
            
            # Evaluate defended model on adversarial examples
            defended_preds = defender.predict(attack_results['adversarial_text'])
            
            # Calculate metrics
            original_accuracy = accuracy_score(attack_results['original_label'], original_preds['predictions'])
            defended_accuracy = accuracy_score(attack_results['original_label'], defended_preds['predictions'])
            
            original_precision, original_recall, original_f1, _ = precision_recall_fscore_support(
                attack_results['original_label'], original_preds['predictions'], average='weighted'
            )
            
            defended_precision, defended_recall, defended_f1, _ = precision_recall_fscore_support(
                attack_results['original_label'], defended_preds['predictions'], average='weighted'
            )
            
            # Calculate attack success rate (lower is better for defense)
            attack_success_rate = sum(attack_results['success']) / len(attack_results['success'])
            
            # Calculate defense efficacy
            defense_efficacy = (defended_accuracy - original_accuracy) / (1 - original_accuracy) if original_accuracy < 1 else 1.0
            
            results['metrics'][attack_method] = {
                'original_accuracy': original_accuracy,
                'defended_accuracy': defended_accuracy,
                'original_precision': original_precision,
                'defended_precision': defended_precision,
                'original_recall': original_recall,
                'defended_recall': defended_recall,
                'original_f1': original_f1,
                'defended_f1': defended_f1,
                'attack_success_rate': attack_success_rate,
                'defense_efficacy': defense_efficacy
            }
            
            # Store examples for visualization
            for i in range(min(10, len(attack_results['original_text']))):
                if attack_results['success'][i]:
                    results['examples'].append({
                        'attack_method': attack_method,
                        'original_text': attack_results['original_text'][i],
                        'adversarial_text': attack_results['adversarial_text'][i],
                        'original_label': attack_results['original_label'][i],
                        'original_model_prediction': original_preds['predictions'][i],
                        'defended_model_prediction': defended_preds['predictions'][i],
                        'original_model_scores': original_preds['scores'][i],
                        'defended_model_scores': defended_preds['scores'][i]
                    })
        
        return results
    
    def visualize_results(self, results):
        """
        Create visualizations for the evaluation results.
        
        Args:
            results (dict): Results from the evaluate method
        """
        # Create plots directory
        import os
        os.makedirs('plots', exist_ok=True)
        
        # Plot 1: Attack Success Rate
        plt.figure(figsize=(10, 6))
        attack_methods = list(results['metrics'].keys())
        attack_success_rates = [results['metrics'][method]['attack_success_rate'] for method in attack_methods]
        
        sns.barplot(x=attack_methods, y=attack_success_rates)
        plt.title('Attack Success Rate by Method')
        plt.xlabel('Attack Method')
        plt.ylabel('Success Rate')
        plt.savefig('plots/attack_success_rate.png')
        
        # Plot 2: Defense Efficacy
        plt.figure(figsize=(10, 6))
        defense_efficacy = [results['metrics'][method]['defense_efficacy'] for method in attack_methods]
        
        sns.barplot(x=attack_methods, y=defense_efficacy)
        plt.title('Defense Efficacy by Attack Method')
        plt.xlabel('Attack Method')
        plt.ylabel('Defense Efficacy')
        plt.savefig('plots/defense_efficacy.png')
        
        # Plot 3: Accuracy Comparison
        plt.figure(figsize=(12, 6))
        
        df = pd.DataFrame({
            'Attack Method': attack_methods + attack_methods,
            'Model': ['Original'] * len(attack_methods) + ['Defended'] * len(attack_methods),
            'Accuracy': [results['metrics'][method]['original_accuracy'] for method in attack_methods] + 
                       [results['metrics'][method]['defended_accuracy'] for method in attack_methods]
        })
        
        sns.barplot(x='Attack Method', y='Accuracy', hue='Model', data=df)
        plt.title('Accuracy Comparison: Original vs. Defended Model')
        plt.xlabel('Attack Method')
        plt.ylabel('Accuracy')
        plt.savefig('plots/accuracy_comparison.png')
        
        # Plot 4: Prediction Changes
        if results['examples']:
            example = results['examples'][0]
            
            plt.figure(figsize=(10, 6))
            
            original_scores = example['original_model_scores']
            defended_scores = example['defended_model_scores']
            
            x = np.arange(len(original_scores))
            width = 0.35
            
            plt.bar(x - width/2, original_scores, width, label='Original Model')
            plt.bar(x + width/2, defended_scores, width, label='Defended Model')
            
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title(f'Prediction Probabilities for Example\nAttack: {example["attack_method"]}')
            plt.xticks(x)
            plt.legend()
            plt.tight_layout()
            plt.savefig('plots/prediction_changes.png')
        
        print(f"Visualizations saved to 'plots/' directory")

