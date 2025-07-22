# Sequence Prediction API Documentation

## Overview
The sequence prediction endpoint provides real-time anomaly detection for log event sequences. It takes a sequence of events and returns whether the sequence indicates an anomaly along with confidence scores.

## Endpoint Details

### POST /predict-sequence/
Predicts anomaly for a single sequence of log events.

**Request Body:**
```json
{
    "event_sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "description": "Sample log event sequence",
    "input_type": "auto"
}
```

For raw log lines:
```json
{
    "event_sequence": [
        "INFO nova.compute.claims [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Claim successful on node parisaserver",
        "INFO nova.virt.libvirt.driver [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Creating image",
        "INFO os_vif [req-a4498d64-47bb-491f-adde-effccaba43f0] Successfully plugged vif VIFOpenVSwitch"
    ],
    "description": "Raw log lines sequence",
    "input_type": "raw_logs"
}
```

**Response:**
```json
{
    "status": "success",
    "is_anomaly": false,
    "confidence": 0.1234,
    "average_score": 0.0987,
    "window_predictions": [false, true],
    "window_scores": [0.1234, 0.5678],
    "sequence_length": 10,
    "windows_analyzed": 2,
    "processed_sequence_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
```

### GET /predict-sequence/info
Returns information about the sequence prediction endpoint including format and examples.

## Request Format

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| event_sequence | List[int or str] | Yes | Sequence of event IDs (integers), event names (strings), or raw log lines |
| description | string | No | Optional description of the sequence |
| input_type | string | No | Type hint: "auto" (default), "event_ids", "event_names", or "raw_logs" |

## Response Format

| Field | Type | Description |
|-------|------|-------------|
| status | string | "success" or "error" |
| is_anomaly | boolean | True if sequence is anomalous, False if normal |
| confidence | float | Anomaly confidence score (0-1, higher = more anomalous) |
| average_score | float | Average anomaly score across all windows |
| window_predictions | List[boolean] | Anomaly predictions for each sliding window |
| window_scores | List[float] | Anomaly scores for each sliding window |
| sequence_length | integer | Length of the processed sequence |
| windows_analyzed | integer | Number of sliding windows analyzed |
| processed_sequence_ids | List[integer] | The event IDs used for prediction (after processing raw logs) |
| message | string | Error message (only present if status is "error") |

## How It Works

1. **Input Processing**: Event sequences can be provided as:
   - Integer IDs (direct event IDs from the vocabulary)
   - String names (converted to IDs using the trained vocabulary)
   - Raw log lines (parsed using Drain algorithm to extract event templates, then mapped to IDs)

2. **Sequence Handling**:
   - Short sequences (< window_size=10): Padded with zeros
   - Long sequences (> window_size=10): Analyzed using sliding windows

3. **Anomaly Detection**:
   - Uses the trained DeepLog LSTM model
   - Predicts the next event for each window
   - Flags anomaly if actual event is not in top candidates
   - Calculates confidence based on prediction probability

4. **Result Aggregation**:
   - Overall anomaly: True if any window shows anomaly
   - Confidence: Maximum anomaly score across windows
   - Average score: Mean anomaly score across windows

## Example Usage

### Normal Sequence
```bash
curl -X POST "http://localhost:8001/predict-sequence/" \
  -H "Content-Type: application/json" \
  -d '{
    "event_sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "description": "Normal sequential events"
  }'
```

### Potentially Anomalous Sequence
```bash
curl -X POST "http://localhost:8001/predict-sequence/" \
  -H "Content-Type: application/json" \
  -d '{
    "event_sequence": [1, 1, 1, 1, 1, 23, 23, 23, 23, 23],
    "description": "Repeated unusual events"
  }'
```

### Using Event Names (if vocabulary supports them)
```bash
curl -X POST "http://localhost:8001/predict-sequence/" \
  -H "Content-Type: application/json" \
  -d '{
    "event_sequence": ["login", "authenticate", "access_denied", "logout"],
    "description": "Authentication sequence with potential issue"
  }'
```

## Error Handling

The endpoint returns appropriate error messages for:
- Empty sequences
- Model loading failures
- Vocabulary loading issues
- Invalid input formats

Example error response:
```json
{
    "status": "error",
    "message": "Event sequence cannot be empty"
}
```

## Requirements

1. **Trained Model**: The DeepLog model must be trained and saved at the configured path
2. **Vocabulary**: The vocabulary file must be available for event name/ID mapping
3. **Dependencies**: FastAPI, PyTorch, and other required libraries

## Testing

Use the provided test scripts:
- `test_sequence_prediction.py` - Python test script (requires `requests` library)
- `test_sequence_prediction.sh` - Bash test script using curl

## Configuration

The endpoint uses configuration from `deeplog.py`:
- Window size: 10 events
- Number of candidates: 9
- Model path: "./output/deeplog/deeplog/bestloss.pth"
- Vocabulary path: "./output/deeplog/vocab.pkl"
