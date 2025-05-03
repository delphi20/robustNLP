import argparse
import os
import sys
import torch
import nltk 
from threat_engine.threat_engine import ThreatGenerator
from defense_engine.defense_engine import Defender
from evaluation.evaluation import Evaluator
from api.api import start_api_server

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description='NLP Defense System Against Adversarial Attacks')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Hugging Face model name to defend')
    parser.add_argument('--dataset', type=str, default='imdb',
                        help='Dataset to use for training and evaluation')
    parser.add_argument('--attack_method', type=str, default='textfooler',
                        choices=['textfooler', 'bert-attack', 'all'],
                        help='Adversarial attack method to use')
    parser.add_argument('--defense_method', type=str, default='adversarial_training',
                        choices=['adversarial_training', 'preprocessing', 'both'],
                        help='Defense method to apply')
    parser.add_argument('--serve_api', action='store_true',
                        help='Start API server for frontend integration')
    parser.add_argument('--api_port', type=int, default=8000,
                        help='Port for API server')
    return parser.parse_args()

def main():
    
    
    nltk.download("stopwords")
    
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize components
    threat_generator = ThreatGenerator(
        model_name=args.model_name,
        attack_method=args.attack_method,
        device=device
    )
    
    defender = Defender(
        model_name=args.model_name,
        defense_method=args.defense_method,
        device=device
    )
    
    evaluator = Evaluator(device=device)
    
    # Load and process dataset
    dataset = defender.load_dataset(args.dataset)
    
    # Training the defended model
    defender.train(dataset, threat_generator)
    
    # Evaluate the original and defended models
    results = evaluator.evaluate(dataset, threat_generator, defender)
    evaluator.visualize_results(results)
    
    # Save the defended model
    defender.save_model(f"defended_{args.model_name}_{args.defense_method}")
    
    # Start API server for frontend integration if requested
    if args.serve_api:
        start_api_server(
            threat_generator=threat_generator,
            defender=defender,
            evaluator=evaluator,
            port=args.api_port
        )

if __name__ == "__main__":
    
    
    
    main()