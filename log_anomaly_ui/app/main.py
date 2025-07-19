
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
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
from uiUtilities import ui_train, ui_data_process, ui_deeplog_predict
app = FastAPI()
# UPLOAD_DIR = "app/uploads"
templates = Jinja2Templates(directory="log_anomaly_ui/app/templates")
# os.makedirs(UPLOAD_DIR, exist_ok=True)

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




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 