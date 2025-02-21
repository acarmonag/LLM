import requests
import json
from typing import Dict, Any
from datetime import datetime

BASE_URL = "http://localhost:8002"

def make_request(endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict:
    """Make HTTP request and handle errors"""
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}/{endpoint}")
        else:
            response = requests.post(
                f"{BASE_URL}/{endpoint}",
                json=data,
                headers={"Content-Type": "application/json"}
            )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error in {endpoint} request:", str(e))
        return {"error": str(e)}

def print_system_info(info: Dict):
    """Print system information in a formatted way"""
    print("\nSystem Information:")
    print(f"CPU Usage: {info['cpu_usage']}")
    print(f"Memory Usage: {info['memory_used']}")
    if isinstance(info['gpu_info'], list):
        gpu = True
        for gpu in info['gpu_info']:
            print(f"\nGPU {gpu['id']}:")
            print(f"  Name: {gpu['name']}")
            print(f"  Load: {gpu['load']}")
            print(f"  Memory: {gpu['memory_used']}/{gpu['memory_total']}")
            print(f"  Temperature: {gpu['temperature']}")
            
    return gpu

def main():
    print(f"\nStarting API tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test system info endpoint
    print("\n1. Testing system info endpoint...")
    result = make_request("system-info")
    gpu = print_system_info(result)

    # Test root endpoint
    print("\n2. Testing root endpoint...")
    result = make_request("")
    print("Root endpoint:", json.dumps(result, indent=2))

    # Test generate endpoint
    print("\n3. Testing generate endpoint...")
    generate_data = {
        "text": "What is Python?",
        "max_length": 100,
        "use_gpu": True
    }
    
    result = make_request("generate", method="POST", data=generate_data)
    print("Generated text:", result.get("generated_text", "No text generated"))
    if "system_info" in result:
        print_system_info(result["system_info"])

    # Test embeddings endpoint
    print("\n4. Testing embeddings endpoint...")
    embedding_data = {
        "texts": ["Hello world", "How are you?"],
        "use_gpu": True
    }
    result = make_request("embeddings", method="POST", data=embedding_data)
    print("Embeddings received:", bool(result.get("embeddings")))
    if "system_info" in result:
        print_system_info(result["system_info"])

if __name__ == "__main__":
    main()