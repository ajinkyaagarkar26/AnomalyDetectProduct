# Sequence Prediction Endpoint - Quick Start

## Summary
I've successfully added a new endpoint `/predict-sequence/` to your anomaly detection API that can handle real-time log sequence analysis.

## Key Features

✅ **Multiple Input Types:**
- Raw log lines (like those in nova-sample.log)
- Event template IDs (numeric)
- Event template names (string)

✅ **Automatic Processing:**
- Raw logs are parsed using the Drain algorithm
- Event templates are mapped to numeric IDs
- Sequences are analyzed using sliding windows

✅ **Real-time Prediction:**
- Uses your trained DeepLog LSTM model
- Returns anomaly probability and confidence scores
- Provides detailed window-by-window analysis

## Quick Usage

### 1. Start the server:
```bash
cd log_anomaly_ui/app
python main.py
```

### 2. Test with raw log lines:
```bash
curl -X POST "http://localhost:8001/predict-sequence/" \
  -H "Content-Type: application/json" \
  -d '{
    "event_sequence": [
      "INFO nova.compute.claims [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Claim successful on node parisaserver",
      "ERROR nova.virt.libvirt.driver [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Failed to create image"
    ],
    "description": "Test sequence with potential error"
  }'
```

### 3. Test with event IDs:
```bash
curl -X POST "http://localhost:8001/predict-sequence/" \
  -H "Content-Type: application/json" \
  -d '{
    "event_sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "description": "Numeric event sequence"
  }'
```

## Response Format
```json
{
  "status": "success",
  "is_anomaly": false,
  "confidence": 0.1234,
  "average_score": 0.0987,
  "sequence_length": 10,
  "windows_analyzed": 2,
  "processed_sequence_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
```

## Files Created/Modified

### New Files:
- `SEQUENCE_PREDICTION_API.md` - Complete API documentation
- `test_sequence_prediction.py` - Python test script
- `test_sequence_prediction.sh` - Bash test script
- `sequence_prediction_examples.py` - Comprehensive examples

### Modified Files:
- `log_anomaly_ui/app/main.py` - Added new endpoint
- `uiUtilities.py` - Added prediction functions

## Testing

Run the test scripts to verify everything works:
```bash
# Make executable and run bash tests
chmod +x test_sequence_prediction.sh
./test_sequence_prediction.sh

# Or run Python tests (requires requests library)
python test_sequence_prediction.py

# Or see examples
python sequence_prediction_examples.py
```

## Integration

The endpoint is now ready for integration into your application. It can handle:
- Real-time log streams
- Batch processing of log sequences  
- Different log formats through the existing Drain parser
- Both normal and anomalous patterns

The system maintains compatibility with your existing training pipeline and uses the same model and vocabulary files.
