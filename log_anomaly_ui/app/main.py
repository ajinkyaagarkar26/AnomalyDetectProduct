
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Union
import shutil, os,sys
# Add the root project directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
# Add the current app directory to Python path for importing config
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(app_dir)
from config import UPLOAD_DIR, TRAIN_DIR
# from deeplog import train
# from data_process_pred import 
# from models.loganomaly import train_loganomaly
from uiUtilities import ui_train, ui_data_process, ui_deeplog_predict, ui_predict_sequence
app = FastAPI()
# UPLOAD_DIR = "app/uploads"
templates = Jinja2Templates(directory="log_anomaly_ui/app/templates")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

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

@app.get("/", response_class=HTMLResponse)
async def show_initial_form(request: Request):
    print("UPLOAD_DIR: ", UPLOAD_DIR)
    return templates.TemplateResponse("index.html", {"request": request, "result": ""})

@app.get("/uploadPage", response_class=HTMLResponse)
async def show_upload_form(request: Request):
    # print("UPLOAD_DIR: ", UPLOAD_DIR)
    return templates.TemplateResponse("upload.html", {"request": request, "result": ""})


@app.post("/trainUpload/")
async def handle_train_upload(request: Request, log_file: UploadFile = File(...), label_file: UploadFile = File(...)):
    print("handle_train_upload UPLOAD_DIR: ", UPLOAD_DIR)
    log_path = os.path.join(UPLOAD_DIR, log_file.filename)
    label_path = os.path.join(UPLOAD_DIR, label_file.filename)

    with open(log_path, "wb") as f1:
        shutil.copyfileobj(log_file.file, f1)
    with open(label_path, "wb") as f2:
        shutil.copyfileobj(label_file.file, f2)

    return templates.TemplateResponse("uploadSuccess.html", {"request": request, "result": ""})

@app.post("/preprocess/")
async def do_preprocess(request: Request, logformat: str = Form(...)):
    print("in do_preprocess:", UPLOAD_DIR)
    preProcResult = ui_train(input_dir=UPLOAD_DIR, output_dir=TRAIN_DIR)
    return {preProcResult}

@app.post("/data-process/")
async def run_data_process(request: Request):
    """
    Endpoint to run the data_process.py pipeline
    Processes logs using the drain parser, mapping, deeplog sampling, and generates train/test data
    """
    print("Starting data processing pipeline")
    try:
        result = ui_data_process()
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/deeplog-predict/")
async def run_deeplog_predict(request: Request):
    """
    Endpoint to run deeplog prediction
    Runs the deeplog prediction pipeline for anomaly detection
    """
    print("Starting deeplog prediction")
    try:
        result = ui_deeplog_predict()
        return {"status": "success", "result": result}
    except Exception as e:
        print(f"Error in deeplog prediction: {str(e)}")
        return {"status": "error", "message": str(e)}

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
        result = ui_predict_sequence(request.event_sequence)
        
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