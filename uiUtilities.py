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

def ui_predict_sequence(event_sequence):
    """
    Predicts anomaly for a single sequence of events
    Args:
        event_sequence: List of event IDs, event names, or raw log lines
    Returns:
        Dictionary with prediction result and confidence
    """
    try:
        print(f"Predicting anomaly for sequence Test: {event_sequence}")
        
        # Import required modules for single sequence prediction
        from deeplog import options
        from logdeep.models.lstm import Deeplog
        from logdeep.tools.predict import Predicter
        import torch
        import pickle
        import numpy as np
        import tempfile
        import os
        # Log only the option parameters used by the model in the next line
        model_keys = ['input_size', 'hidden_size', 'num_layers', 'vocab_size', 'embedding_dim', 'model_path', 'device']
        print("Deeplog model options used for model initialization:")
        for k in model_keys:
            print(f"  {k}: {options.get(k)}")
        print(f"Using model path: {options['model_path']}")
        # Load the trained model
        model = Deeplog(options['input_size'], 
                     options['hidden_size'], 
                     options['num_layers'], 
                     options["vocab_size"],
                     options["embedding_dim"])
        
        model.load_state_dict(torch.load(options['model_path'])['state_dict'])
        model.eval()
        model = model.to(options['device'])
        
        # Load vocabulary
        with open(options['vocab_path'], 'rb') as f:
            vocab = pickle.load(f)
        
        # Process input sequence based on type
        sequence_ids = []
        
        # Check if input contains raw log lines (strings with spaces/formatting)
        if isinstance(event_sequence[0], str) and any(len(event.split()) > 2 for event in event_sequence):
            # Raw log lines - need to parse them
            sequence_ids = _parse_log_sequence_to_ids(event_sequence)
        elif isinstance(event_sequence[0], str):
            # Event template names - convert using vocab
            try:
                sequence_ids = [vocab.stoi.get(event, vocab.stoi.get('<UNK>', 0)) for event in event_sequence]
            except AttributeError:
                # Handle different vocab structure
                sequence_ids = [vocab.get(event, 0) for event in event_sequence]
        else:
            # Already numeric IDs
            sequence_ids = event_sequence
        
        # Ensure sequence has minimum length
        if len(sequence_ids) < options['window_size']:
            # Pad with zeros if sequence is too short
            sequence_ids = [0] * (options['window_size'] - len(sequence_ids)) + sequence_ids
        
        # Take sliding window approach for sequences longer than window size
        predictions = []
        anomaly_scores = []
        
        for i in range(len(sequence_ids) - options['window_size'] + 1):
            window = sequence_ids[i:i + options['window_size']]
            
            # Convert to tensor
            input_seq = torch.tensor(window[:-1], dtype=torch.long).unsqueeze(0).to(options['device'])
            target = torch.tensor([window[-1]], dtype=torch.long).to(options['device'])
            
            with torch.no_grad():
                output = model([input_seq], options['device'])
                
                # Get top candidates for prediction
                predicted_candidates = torch.argsort(output, descending=True)[0][:options['num_candidates']]
                
                # Check if actual next event is in top candidates
                is_anomaly = target[0] not in predicted_candidates
                
                # Calculate anomaly score (inverse of prediction probability)
                probs = torch.softmax(output, dim=1)
                target_prob = probs[0][target[0]].item()
                anomaly_score = 1 - target_prob
                
                predictions.append(is_anomaly)
                anomaly_scores.append(anomaly_score)
        
        # Overall prediction - if any window shows anomaly
        overall_anomaly = any(predictions)
        avg_anomaly_score = np.mean(anomaly_scores) if anomaly_scores else 0
        max_anomaly_score = max(anomaly_scores) if anomaly_scores else 0
        
        return {
            "status": "success",
            "is_anomaly": overall_anomaly,
            "confidence": float(max_anomaly_score),
            "average_score": float(avg_anomaly_score),
            "window_predictions": predictions,
            "window_scores": anomaly_scores,
            "sequence_length": len(sequence_ids),
            "windows_analyzed": len(predictions),
            "processed_sequence_ids": sequence_ids
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Sequence prediction failed: {str(e)}"
        }


def _parse_log_sequence_to_ids(log_lines):
    """
    Parse a sequence of raw log lines to event IDs using the Drain parser
    Args:
        log_lines: List of raw log lines
    Returns:
        List of event numeric IDs
    """
    try:
        import tempfile
        import os
        import pandas as pd
        from logparser.Drain import LogParser
        
        # Create temporary files for processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as temp_log:
            for line in log_lines:
                temp_log.write(line.strip() + '\n')
            temp_log_path = temp_log.name
        
        temp_output_dir = tempfile.mkdtemp()
        
        try:
            # Initialize Drain parser with same settings as training
            log_format = '<Level> <Component> <ADDR> <Content>'
            parser = LogParser(log_format, indir=os.path.dirname(temp_log_path), 
                             outdir=temp_output_dir, depth=5, st=0.4)
            
            # Parse the logs
            log_file = os.path.basename(temp_log_path)
            parser.parse(log_file)
            
            # Read the structured output
            structured_file = os.path.join(temp_output_dir, log_file + '_structured.csv')
            if os.path.exists(structured_file):
                df = pd.read_csv(structured_file)
                
                # Load the existing templates to get EventNumericId mapping
                templates_file = './output/deeplog/nova-sample.log_templates.csv'
                if os.path.exists(templates_file):
                    templates_df = pd.read_csv(templates_file)
                    template_to_id = dict(zip(templates_df['EventId'], templates_df['templateIndex']))
                    
                    # Map EventIds to numeric IDs
                    event_ids = []
                    for event_id in df['EventId']:
                        numeric_id = template_to_id.get(event_id, 0)  # Default to 0 for unknown
                        event_ids.append(numeric_id)
                    
                    return event_ids
                else:
                    # Fallback: use EventId hash as numeric representation
                    from deeplog import options
                    return [hash(event_id) % options['vocab_size'] for event_id in df['EventId']]
            else:
                raise Exception("Failed to parse log lines")
                
        finally:
            # Cleanup temp files
            if os.path.exists(temp_log_path):
                os.unlink(temp_log_path)
            import shutil
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
    
    except Exception as e:
        print(f"Error parsing log lines: {str(e)}")
        # Fallback: return simple hash-based IDs
        return [hash(line) % 100 for line in log_lines]

def ui_predict():
    print("hello world")
    return


if __name__ == "__main__":    
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))