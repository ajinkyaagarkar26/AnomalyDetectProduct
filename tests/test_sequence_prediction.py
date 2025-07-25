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
        {
            "name": "Raw log lines sequence",
            "data": {
                "event_sequence": [
                    "INFO nova.compute.claims [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Claim successful on node parisaserver",
                    "INFO nova.virt.libvirt.driver [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Creating image",
                    "INFO os_vif [req-a4498d64-47bb-491f-adde-effccaba43f0] Successfully plugged vif VIFOpenVSwitch",
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 11760334-ac63-4cc8-9086-578422af8c99] VM Started (Lifecycle Event)",
                    "INFO nova.virt.libvirt.driver [-] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Instance spawned successfully.",
                    "INFO nova.compute.manager [req-a4498d64-47bb-491f-adde-effccaba43f0] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Took 9.31 seconds to spawn the instance on the hypervisor.",
                    "INFO nova.compute.manager,[req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53],[instance: 11760334-ac63-4cc8-9086-578422af8c99] VM Paused (Lifecycle Event)",
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 11760334-ac63-4cc8-9086-578422af8c99] VM Resumed (Lifecycle Event)",
                    "WARNING nova.virt.libvirt.driver [req-32833659-9280-430c-b915-a246f788ad28] Error from libvirt while getting description of instance-00000072",
                    "ERROR nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Unknown error occurred"
                ],
                "description": "Raw log lines with potential anomaly",
                "input_type": "raw_logs"
            }
        },
        {
            "name": "Real nova log sequence from sample file",
            "data": {
                "event_sequence": [
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 2e0f047e-8d2f-4166-a950-ecf24c8743a7] VM Started (Lifecycle Event)",
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 2e0f047e-8d2f-4166-a950-ecf24c8743a7] VM Paused (Lifecycle Event)",
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 2e0f047e-8d2f-4166-a950-ecf24c8743a7] During sync_power_state the instance has a pending task (spawning). Skip.",
                    "INFO nova.virt.libvirt.driver [req-b0f9d3a5-273c-401d-9c61-f84cf0a91b18] [instance: 2e0f047e-8d2f-4166-a950-ecf24c8743a7] Deleting instance files /opt/stack/data/nova/instances/2e0f047e-8d2f-4166-a950-ecf24c8743a7_del",
                    "INFO nova.virt.libvirt.driver [req-b0f9d3a5-273c-401d-9c61-f84cf0a91b18] [instance: 2e0f047e-8d2f-4166-a950-ecf24c8743a7] Deletion of /opt/stack/data/nova/instances/2e0f047e-8d2f-4166-a950-ecf24c8743a7_del complete",                                        
                    "INFO nova.compute.manager [req-ee9a4f4d-45b6-47cf-9541-92f8433b3592] [instance: 2e0f047e-8d2f-4166-a950-ecf24c8743a7] Took 58.84 seconds to build instance.",
                    "INFO nova.compute.manager [req-b0f9d3a5-273c-401d-9c61-f84cf0a91b18] [instance: 2e0f047e-8d2f-4166-a950-ecf24c8743a7] Terminating instance",
                    "INFO nova.compute.manager [req-b0f9d3a5-273c-401d-9c61-f84cf0a91b18] [instance: 2e0f047e-8d2f-4166-a950-ecf24c8743a7] Took 4.51 seconds to destroy the instance on the hypervisor.",
                    "INFO nova.compute.manager [req-37b90fcc-54bf-416b-a83c-8a6c5f6cb55d req-6c5f6906-0acc-490e-ba57-44946ef7c57d service nova] [instance: 2e0f047e-8d2f-4166-a950-ecf24c8743a7] Neutron deleted interface 1e0088b3-9b3f-45e6-aeba-639a7208fdc8; detaching it from the instance and deleting it from the info cache",
                    "INFO nova.compute.claims [req-ee9a4f4d-45b6-47cf-9541-92f8433b3592] [instance: 2e0f047e-8d2f-4166-a950-ecf24c8743a7] Claim successful on node parisaserver"
                ],
                "description": "Real log sequence from nova-sample.log showing instance lifecycle",
                "input_type": "raw_logs"
            }
        },
        {
            "name": "Normal instance lifecycle",
            "data": {
                "event_sequence": [
                    "INFO nova.compute.claims [req-e665dbf5-dcc9-468d-bc98-eb9aaf8071c6] [instance: b9f0f7cd-31c1-4edf-9a8d-129d4efff387] Claim successful on node parisaserver",
                    "INFO nova.virt.libvirt.driver [req-e665dbf5-dcc9-468d-bc98-eb9aaf8071c6] [instance: b9f0f7cd-31c1-4edf-9a8d-129d4efff387] Creating image",
                    "INFO os_vif [req-e665dbf5-dcc9-468d-bc98-eb9aaf8071c6] Successfully plugged vif VIFOpenVSwitch",
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: b9f0f7cd-31c1-4edf-9a8d-129d4efff387] VM Started (Lifecycle Event)",
                    "INFO nova.virt.libvirt.driver [-] [instance: b9f0f7cd-31c1-4edf-9a8d-129d4efff387] Instance spawned successfully.",
                    "INFO nova.compute.manager [req-e665dbf5-dcc9-468d-bc98-eb9aaf8071c6] [instance: b9f0f7cd-31c1-4edf-9a8d-129d4efff387] Took 9.31 seconds to spawn the instance on the hypervisor."
                ],
                "description": "Normal instance creation and startup sequence",
                "input_type": "raw_logs"
            }
        },
        {
            "name": "Instance termination sequence - Non Anomaly",
            "data": {
                "event_sequence": [
                    "INFO nova.compute.manager [req-979f4d38-e788-4abf-ab98-5d63ce15f517] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Terminating instance",
                    "INFO nova.virt.libvirt.driver [-] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Instance destroyed successfully.",
                    "INFO os_vif [req-979f4d38-e788-4abf-ab98-5d63ce15f517] Successfully unplugged vif VIFOpenVSwitch",
                    "INFO nova.virt.libvirt.driver [req-979f4d38-e788-4abf-ab98-5d63ce15f517] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Deleting instance files /opt/stack/data/nova/instances/11760334-ac63-4cc8-9086-578422af8c99_del",
                    "INFO nova.compute.manager [req-979f4d38-e788-4abf-ab98-5d63ce15f517] [instance: 11760334-ac63-4cc8-9086-578422af8c99] Took 4.14 seconds to destroy the instance on the hypervisor.",
                    "INFO nova.scheduler.client.report [req-979f4d38-e788-4abf-ab98-5d63ce15f517] Deleted allocation for instance 11760334-ac63-4cc8-9086-578422af8c99"
                ],
                "description": "Normal instance termination and cleanup sequence",
                "input_type": "raw_logs"
            }
        },
        {
            "name": "VM lifecycle with pause resume",
            "data": {
                "event_sequence": [
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 506a23f5-6cec-4d64-abf8-16e017154255] VM Started (Lifecycle Event)",
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 506a23f5-6cec-4d64-abf8-16e017154255] VM Paused (Lifecycle Event)",
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 506a23f5-6cec-4d64-abf8-16e017154255] During sync_power_state the instance has a pending task (spawning). Skip.",
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 506a23f5-6cec-4d64-abf8-16e017154255] VM Resumed (Lifecycle Event)",
                    "INFO nova.virt.libvirt.driver [-] [instance: 506a23f5-6cec-4d64-abf8-16e017154255] Instance spawned successfully.",
                    "INFO nova.compute.manager [req-01e56c59-0677-42e5-ba82-405497483d8e] [instance: 506a23f5-6cec-4d64-abf8-16e017154255] Took 82.50 seconds to spawn the instance on the hypervisor."
                ],
                "description": "VM lifecycle with pause and resume events",
                "input_type": "raw_logs"
            }
        },
        {
            "name": "Libvirt domain error sequence - Non Anomaly",
            "data": {
                "event_sequence": [
                    "INFO nova.compute.manager [req-2d399014-32df-490c-97a4-646ba82df5cf] [instance: f52f599a-b2a2-478c-bcfe-9933528dd526] Terminating instance",
                    "INFO nova.virt.libvirt.driver [-] [instance: f52f599a-b2a2-478c-bcfe-9933528dd526] Instance destroyed successfully.",
                    "INFO nova.compute.manager [req-2d399014-32df-490c-97a4-646ba82df5cf] [instance: f52f599a-b2a2-478c-bcfe-9933528dd526] Took 3.86 seconds to destroy the instance on the hypervisor.",
                    "WARNING nova.virt.libvirt.driver [req-32833659-9280-430c-b915-a246f788ad28] Error from libvirt while getting description of instance-00000072: [Error Code 42] Domain not found: no domain with matching uuid 'f52f599a-b2a2-478c-bcfe-9933528dd526' (instance-00000072): libvirt.libvirtError: Domain not found",
                    "INFO nova.compute.manager [-] [instance: f52f599a-b2a2-478c-bcfe-9933528dd526] VM Stopped (Lifecycle Event)",
                    "INFO nova.scheduler.client.report [req-2d399014-32df-490c-97a4-646ba82df5cf] Deleted allocation for instance f52f599a-b2a2-478c-bcfe-9933528dd526"
                ],
                "description": "Instance termination with libvirt domain not found warning",
                "input_type": "raw_logs"
            }
        },
        {
            "name": "Power state synchronization warning",
            "data": {
                "event_sequence": [
                    "INFO nova.compute.manager [req-77396c32-0c2f-4d85-bb19-7174aa911288] [instance: 2f093863-6416-4d09-ac00-7ea45d932d36] Took 82.81 seconds to spawn the instance on the hypervisor.",
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 2f093863-6416-4d09-ac00-7ea45d932d36] VM Resumed (Lifecycle Event)",
                    "INFO nova.compute.manager [req-77396c32-0c2f-4d85-bb19-7174aa911288] [instance: 2f093863-6416-4d09-ac00-7ea45d932d36] Took 93.32 seconds to build instance.",
                    "WARNING nova.compute.manager [req-32833659-9280-430c-b915-a246f788ad28] While synchronizing instance power states, found 9 instances in the database and 8 instances on the hypervisor.",
                    "INFO nova.virt.libvirt.driver [req-e9767169-a830-4f59-a660-0d28173bec12] [instance: e4faecd7-54e4-470b-83af-8c021fb864d7] Creating image"
                ],
                "description": "Power state synchronization issue between database and hypervisor",
                "input_type": "raw_logs"
            }
        },
        {
            "name": "Disk storage error sequence",
            "data": {
                "event_sequence": [
                    "INFO nova.compute.manager [req-ac9f5721-5c52-4ec3-ba8a-e494d9780d53] [instance: 8505e839-8c3f-40a0-8eff-41c36ba17e9d] VM Started (Lifecycle Event)",
                    "INFO nova.virt.libvirt.driver [-] [instance: 8505e839-8c3f-40a0-8eff-41c36ba17e9d] Instance spawned successfully.",
                    "WARNING nova.virt.libvirt.driver [req-32833659-9280-430c-b915-a246f788ad28] Periodic task is updating the host stats, it is trying to get disk info for instance-00000090, but the backing disk storage was removed by a concurrent operation such as resize. Error: No disk at /opt/stack/data/nova/instances/8505e839-8c3f-40a0-8eff-41c36ba17e9d/disk: nova.exception.DiskNotFound",
                    "WARNING nova.compute.manager [req-32833659-9280-430c-b915-a246f788ad28] While synchronizing instance power states, found 9 instances in the database and 7 instances on the hypervisor.",
                    "INFO nova.compute.manager [req-1234abcd-5678-90ef-abcd-1234567890ab] [instance: 8505e839-8c3f-40a0-8eff-41c36ba17e9d] Terminating instance"
                ],
                "description": "Disk storage issue during host stats update with power state mismatch",
                "input_type": "raw_logs"
            }
        }
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
