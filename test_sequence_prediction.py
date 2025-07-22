#!/usr/bin/env python3
"""
Test script for the sequence prediction endpoint
"""
import requests
import json

# Configuration
BASE_URL = "http://localhost:8001"
ENDPOINT = "/predict-sequence/"

def test_sequence_prediction():
    """Test the sequence prediction endpoint with sample data"""
    
    # Test cases
    test_cases = [
        # {
        #     "name": "Normal sequence with integers",
        #     "data": {
        #         "event_sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #         "description": "Normal sequential events"
        #     }
        # },
        # {
        #     "name": "Short sequence",
        #     "data": {
        #         "event_sequence": [1, 2, 3],
        #         "description": "Short sequence - should be padded"
        #     }
        # },
        # {
        #     "name": "Long sequence",
        #     "data": {
        #         "event_sequence": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #         "description": "Long sequence - should use sliding windows"
        #     }
        # },
        # {
        #     "name": "Potentially anomalous sequence",
        #     "data": {
        #         "event_sequence": [1, 1, 1, 1, 1, 23, 23, 23, 23, 23],
        #         "description": "Repeated unusual events"
        #     }
        # },
        {
            "name": "Raw log lines sequence",
            "data": {
                "event_sequence": [
                    "INFO nova.compute.claims [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Claim successful on node parisaserver",
                    "INFO nova.virt.libvirt.driver [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Creating image",
                    "INFO os_vif [req-a4498d64-47bb-491f-adde-effccaba43f0] Successfully plugged vif VIFOpenVSwitch",
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 11760334-ac63-4cc8-9086-578422af8c99] VM Started (Lifecycle Event)",
                    "ERROR nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Unknown error occurred"
                ],
                "description": "Raw log lines with potential anomaly",
                "input_type": "raw_logs"
            }
        }
        # ,
        # {
        #     "name": "Empty sequence (should fail)",
        #     "data": {
        #         "event_sequence": [],
        #         "description": "Empty sequence test"
        #     }
        # }
    ]
    
    print(f"Testing sequence prediction endpoint at {BASE_URL}{ENDPOINT}")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Input: {test_case['data']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}{ENDPOINT}",
                json=test_case['data'],
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Status Code: {response.status_code}")
                print(f"Response: {json.dumps(result, indent=2)}")
                
                if result.get("status") == "success":
                    print(f"✓ Anomaly Detected: {result.get('is_anomaly', 'N/A')}")
                    print(f"✓ Confidence: {result.get('confidence', 'N/A'):.4f}" if result.get('confidence') else "✓ Confidence: N/A")
                    print(f"✓ Windows Analyzed: {result.get('windows_analyzed', 'N/A')}")
                else:
                    print(f"✗ Error: {result.get('message', 'Unknown error')}")
            else:
                print(f"✗ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("✗ Connection Error: Make sure the server is running on localhost:8001")
        except Exception as e:
            print(f"✗ Error: {str(e)}")
        
        print("-" * 40)

def test_info_endpoint():
    """Test the info endpoint"""
    print(f"\nTesting info endpoint at {BASE_URL}/predict-sequence/info")
    
    try:
        response = requests.get(f"{BASE_URL}/predict-sequence/info")
        if response.status_code == 200:
            info = response.json()
            print("✓ Info endpoint working")
            print(json.dumps(info, indent=2))
        else:
            print(f"✗ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    print("Sequence Prediction API Test")
    print("=" * 60)
    
    # Test info endpoint first
    #test_info_endpoint()
    
    # Test prediction endpoint
    test_sequence_prediction()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("\nTo run the server, use:")
    print("cd log_anomaly_ui/app && python main.py")
