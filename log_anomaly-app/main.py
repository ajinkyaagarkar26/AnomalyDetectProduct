
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import os,sys
# Add the root project directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
# Add the current app directory to Python path for importing config
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(app_dir)
from fastapi.middleware.cors import CORSMiddleware
from log_anomaly_service import predict_sequence
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your list of allowed origins
    allow_credentials=True,  # Allow cookies and authorization headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Pydantic models for request/response
class SequencePredictionRequest(BaseModel):
    event_sequence: List[Union[int, str]]
    description: str = "Event sequence for anomaly detection"
    input_type: str = "auto"  # "auto", "event_ids", "event_names", "raw_logs"

class SequencePredictionResponse(BaseModel):
    status: str
    is_anomaly: bool = None
    confidence: float = None
    average_score: float = None
    window_predictions: List[bool] = None
    window_scores: List[float] = None
    sequence_length: int = None
    windows_analyzed: int = None
    processed_sequence_ids: List[int] = None
    message: str = None
    

@app.post("/predict-sequence/", response_model=SequencePredictionResponse)
async def predict_sequence_anomaly(request: SequencePredictionRequest):
    """
    Endpoint to predict anomaly for a single sequence of events
    
    Args:
        request: SequencePredictionRequest containing event_sequence (list of event IDs or names)
    
    Returns:
        SequencePredictionResponse with anomaly prediction results
    
    Example:
        POST /predict-sequence/
        {
            "event_sequence": [1, 2, 3, 4, 5],
            "description": "Sample log event sequence"
        }
    """
    print(f"Received sequence prediction request: {request.event_sequence}")
    
    try:
        # Validate input
        if not request.event_sequence or len(request.event_sequence) == 0:
            return SequencePredictionResponse(
                status="error",
                message="Event sequence cannot be empty"
            )
        
        # Call the prediction function
        result = predict_sequence(request.event_sequence)
        
        if result["status"] == "success":
            return SequencePredictionResponse(
                status=result["status"],
                is_anomaly=result["is_anomaly"],
                confidence=result["confidence"],
                average_score=result["average_score"],
                window_predictions=result["window_predictions"],
                window_scores=result["window_scores"],
                sequence_length=result["sequence_length"],
                windows_analyzed=result["windows_analyzed"],
                processed_sequence_ids=result.get("processed_sequence_ids", [])
            )
        else:
            return SequencePredictionResponse(
                status=result["status"],
                message=result["message"]
            )
            
    except Exception as e:
        print(f"Error in sequence prediction: {str(e)}")
        return SequencePredictionResponse(
            status="error",
            message=f"Sequence prediction failed: {str(e)}"
        )

@app.get("/predict-sequence/info")
async def get_sequence_prediction_info():
    """
    Get information about the sequence prediction endpoint
    """
    return {
        "endpoint": "/predict-sequence/",
        "method": "POST",
        "description": "Predicts anomaly for a sequence of log events",
        "input_format": {
            "event_sequence": "List of event IDs (integers) or event names (strings)",
            "description": "Optional description of the sequence"
        },
        "output_format": {
            "status": "success or error",
            "is_anomaly": "Boolean indicating if sequence is anomalous",
            "confidence": "Anomaly confidence score (0-1, higher = more anomalous)",
            "average_score": "Average anomaly score across all windows",
            "window_predictions": "List of anomaly predictions for each window",
            "window_scores": "List of anomaly scores for each window",
            "sequence_length": "Length of the processed sequence",
            "windows_analyzed": "Number of sliding windows analyzed"
        },
        "example_request": {
            "event_sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "description": "Sample sequence of log events"
        },
        "notes": [
            "Sequences shorter than window_size (10) will be padded",
            "Sequences longer than window_size will be analyzed using sliding windows",
            "Event names will be converted to IDs using the trained vocabulary",
            "Unknown events will be mapped to ID 0 (UNK token)"
        ]
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 