#!/usr/bin/env python3
"""
Comprehensive examples for the sequence prediction endpoint showing different input types
"""

# Example raw log lines from nova-sample.log
raw_log_examples = [
    # Normal sequence
    [
        "INFO nova.compute.claims [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Claim successful on node parisaserver",
        "INFO nova.virt.libvirt.driver [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Creating image", 
        "INFO os_vif [req-a4498d64-47bb-491f-adde-effccaba43f0] Successfully plugged vif VIFOpenVSwitch",
        "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 11760334-ac63-4cc8-9086-578422af8c99] VM Started (Lifecycle Event)",
        "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 11760334-ac63-4cc8-9086-578422af8c99] VM Paused (Lifecycle Event)",
    ],
    
    # Potentially anomalous sequence with error
    [
        "INFO nova.compute.claims [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Claim successful on node parisaserver",
        "ERROR nova.virt.libvirt.driver [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Failed to create image",
        "ERROR nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Build failed with unknown error",
        "ERROR nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Instance termination initiated",
    ]
]

# Example event ID sequences (based on templateIndex from nova-sample.log_templates.csv)
event_id_examples = [
    # Normal sequence (common operations)
    [10, 11, 18, 2, 3, 1, 14, 17, 16],  # Claim -> Create -> Plug -> Start -> Pause -> Sync -> Resume -> Spawn -> Complete
    
    # Potentially anomalous sequence 
    [10, 11, 99, 99, 99],  # Normal start then unknown events
    
    # Short sequence
    [2, 5, 6],  # Start -> Stop -> Destroy
]

# Example requests for the API
api_examples = [
    {
        "name": "Raw log lines - normal sequence",
        "request": {
            "event_sequence": raw_log_examples[0],
            "description": "Normal VM creation sequence from raw logs",
            "input_type": "raw_logs"
        }
    },
    {
        "name": "Raw log lines - anomalous sequence", 
        "request": {
            "event_sequence": raw_log_examples[1],
            "description": "VM creation with errors from raw logs",
            "input_type": "raw_logs"
        }
    },
    {
        "name": "Event IDs - normal sequence",
        "request": {
            "event_sequence": event_id_examples[0],
            "description": "Normal VM lifecycle using event IDs",
            "input_type": "event_ids"
        }
    },
    {
        "name": "Event IDs - anomalous sequence",
        "request": {
            "event_sequence": event_id_examples[1], 
            "description": "Normal start followed by unknown events",
            "input_type": "event_ids"
        }
    },
    {
        "name": "Event IDs - short sequence",
        "request": {
            "event_sequence": event_id_examples[2],
            "description": "Short VM lifecycle sequence",
            "input_type": "event_ids"
        }
    }
]

if __name__ == "__main__":
    import json
    
    print("=== Sequence Prediction API Examples ===\n")
    
    for i, example in enumerate(api_examples, 1):
        print(f"Example {i}: {example['name']}")
        print("Request:")
        print(json.dumps(example['request'], indent=2))
        print("\nCurl command:")
        print(f"curl -X POST 'http://localhost:8001/predict-sequence/' \\")
        print(f"  -H 'Content-Type: application/json' \\")
        print(f"  -d '{json.dumps(example['request'])}'")
        print("\n" + "="*60 + "\n")
    
    print("Notes:")
    print("- Raw log lines are automatically parsed using the Drain algorithm")
    print("- Event IDs correspond to templateIndex from the training data")
    print("- Short sequences are padded, long sequences use sliding windows")
    print("- The API auto-detects input type but you can specify it explicitly")
