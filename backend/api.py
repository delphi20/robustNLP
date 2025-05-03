import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class InputText(BaseModel):
    text: str
    
class BatchTexts(BaseModel):
    texts: List[str]
    
class GenerateAttackRequest(BaseModel):
    texts: List[str]
    labels: List[int]
    attack_method: str = "textfooler"

class DefendRequest(BaseModel):
    texts: List[str]
    use_preprocessing: bool = True

class EvaluationRequest(BaseModel):
    dataset_name: str = "imdb"
    num_samples: int = 50
    
class PlotRequest(BaseModel):
    plot_type: str
    data: Dict[str, Any]

def start_api_server(threat_generator, defender, evaluator, port=8000):
    """
    Start a FastAPI server for frontend integration.
    
    Args:
        threat_generator: ThreatGenerator instance
        defender: Defender instance
        evaluator: Evaluator instance
        port (int): Port to run the server on
    """
    app = FastAPI(title="NLP Defense System API")
    
    # Add CORS middleware for Next.js frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For development; restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define the routes
    @app.get("/")
    def read_root():
        return {"message": "NLP Defense System API is running"}
    
    @app.get("/attacks")
    def get_attack_methods():
        return {"attack_methods": threat_generator.get_available_attack_methods()}
    
    @app.post("/generate-attack")
    def generate_attack(request: GenerateAttackRequest):
        try:
            results = threat_generator.generate_adversarial_examples(
                request.texts, request.labels, attack_method=request.attack_method
            )
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/defend")
    def defend_text(request: DefendRequest):
        try:
            # Get predictions from both original and defended models
            original_preds = defender.predict_original_model(request.texts)
            defended_preds = defender.predict(request.texts, use_preprocessing=request.use_preprocessing)
            
            # Combine results
            results = {
                "original_predictions": original_preds["predictions"],
                "defended_predictions": defended_preds["predictions"],
                "original_scores": original_preds["scores"],
                "defended_scores": defended_preds["scores"],
            }
            
            if request.use_preprocessing:
                results["preprocessed_texts"] = [defender.preprocess_text(text) for text in request.texts]
            
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/evaluate")
    def evaluate_system(request: EvaluationRequest):
        try:
            # Load dataset
            dataset = defender.load_dataset(request.dataset_name)
            
            # Run evaluation
            results = evaluator.evaluate(dataset, threat_generator, defender, num_samples=request.num_samples)
            
            # Convert numpy values to Python native types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                else:
                    return obj
            
            results = convert_numpy(results)
            return results
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/plot")
    def generate_plot(request: PlotRequest):
        try:
            plt.figure(figsize=(10, 6))
            
            if request.plot_type == "bar":
                labels = request.data.get("labels", [])
                values = request.data.get("values", [])
                title = request.data.get("title", "Bar Plot")
                
                sns.barplot(x=labels, y=values)
                plt.title(title)
                plt.xlabel(request.data.get("xlabel", ""))
                plt.ylabel(request.data.get("ylabel", ""))
            
            elif request.plot_type == "comparison":
                df = pd.DataFrame(request.data.get("data", []))
                x = request.data.get("x", "x")
                y = request.data.get("y", "y")
                hue = request.data.get("hue", None)
                title = request.data.get("title", "Comparison Plot")
                
                sns.barplot(x=x, y=y, hue=hue, data=df)
                plt.title(title)
                plt.xlabel(request.data.get("xlabel", ""))
                plt.ylabel(request.data.get("ylabel", ""))
            
            # Save plot to a bytes buffer
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            
            # Convert to base64 for sending to frontend
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            return {"plot_data": plot_data}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=port)
    