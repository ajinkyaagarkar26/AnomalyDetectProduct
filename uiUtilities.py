from  data_process_train import parser, mapping, deeplog_sampling, generate_train_test
import os
import sys

# Add imports for data processing and prediction
from data_process import parser as data_process_parser, mapping as data_process_mapping, deeplog_sampling as data_process_deeplog_sampling, generate_train_test as data_process_generate_train_test, run_data_processing_pipeline
from data_process_pred import prePredict_process, parser as pred_parser
from deeplog import predict, process_vocab

def ui_train(input_dir='./trainInput/', output_dir='./trainOutput/', log_file="nova-sample-training.log", log_format='<Level> <Component> <ADDR> <Content>'):

    print("in ui_train, input_dir:",str(input_dir))
    parseResult = parser(input_dir, output_dir, log_file,log_format)
    # print("in ui_train:",str(output_dir) + "/event_sequence.csv")
    trainResult = generate_train_test(str(output_dir) + "/event_sequence.csv")
    return {parseResult, trainResult}

def ui_data_process():
    """
    Runs the complete data processing pipeline from data_process.py
    """
    try:
        print("Starting data processing pipeline via UI")
        
        # Run the complete data processing pipeline
        result = run_data_processing_pipeline()
        
        return {
            "status": "success",
            "message": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Data processing failed: {str(e)}"
        }

def ui_deeplog_predict(pred_input_dir='./predInput/', pred_output_dir='./predOutput/', log_file="nova-sample.log", log_format='<Level> <Component> <ADDR> <Content>'):
    """
    Runs the deeplog prediction pipeline
    """
    try:
        print(f"Starting deeplog prediction with pred_input_dir: {pred_input_dir}")
        
        # Preprocess for prediction
        #preprocess_result = prePredict_process()
        
        # Run prediction parser
        #parser_result = pred_parser(pred_input_dir, pred_output_dir, log_file, log_format)
        
        # Process vocab for prediction
        from deeplog import options
        #vocab_result = process_vocab(options)
        
        # Run prediction
        prediction_result = predict()
        
        return {
            "status": "success", 
            "message": "Deeplog prediction completed successfully",
            #"preprocessResult": preprocess_result,
            #"parserResult": parser_result,
            #"vocabResult": vocab_result,
            "predictionResult": prediction_result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Deeplog prediction failed: {str(e)}"
        }

def ui_predict():
    print("hello world")
    return


if __name__ == "__main__":    
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))