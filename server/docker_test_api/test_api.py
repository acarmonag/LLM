import requests
import json
from typing import Dict, Any
from datetime import datetime
import os
import time
import sys

class APITester:
    def __init__(self):
        self.base_url = os.getenv('API_URL', 'http://localhost:8002')
        self.success_count = 0
        self.fail_count = 0

    def make_request(self, endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict:
        """Make HTTP request with retry mechanism"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}/{endpoint}")
                else:
                    response = requests.post(
                        f"{self.base_url}/{endpoint}",
                        json=data,
                        headers={"Content-Type": "application/json"}
                    )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Error in {endpoint} request after {max_retries} attempts:", str(e))
                    return {"error": str(e)}
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    def print_test_result(self, test_name: str, success: bool, details: str = ""):
        """Print formatted test results"""
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"\n{status} - {test_name}")
        if details:
            print(f"Details: {details}")
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1

    def test_health_check(self):
        """Test API health check endpoint"""
        result = self.make_request("")
        success = isinstance(result, dict) and "message" in result
        self.print_test_result(
            "Health Check",
            success,
            json.dumps(result, indent=2) if success else "Failed to get valid response"
        )

    def test_system_info(self):
        """Test system information endpoint"""
        result = self.make_request("system-info")
        success = isinstance(result, dict) and "cpu_usage" in result
        self.print_test_result(
            "System Info",
            success,
            f"CPU: {result.get('cpu_usage', 'N/A')}, Memory: {result.get('memory_used', 'N/A')}" if success else "Failed to get system info"
        )

    def test_text_generation(self, prompts: list[str]):
        """Test text generation with multiple prompts"""
        for prompt in prompts:
            data = {
                "text": prompt,
                "max_length": 100,
                "use_gpu": False
            }
            result = self.make_request("generate", method="POST", data=data)
            success = isinstance(result, dict) and "generated_text" in result
            self.print_test_result(
                f"Text Generation - '{prompt[:30]}...'",
                success,
                result.get("generated_text", "")[:100] + "..." if success else "Failed to generate text"
            )

    def test_embeddings(self):
        """Test embeddings generation"""
        data = {
            "texts": ["Test embedding", "Another test"],
            "use_gpu": False
        }
        result = self.make_request("embeddings", method="POST", data=data)
        success = isinstance(result, dict) and "embeddings" in result
        self.print_test_result(
            "Embeddings Generation",
            success,
            f"Generated {len(result.get('embeddings', []))} embeddings" if success else "Failed to generate embeddings"
        )

    def run_all_tests(self):
        """Run all API tests"""
        print(f"\n🚀 Starting API tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Testing API at: {self.base_url}\n")
        
        self.test_health_check()
        self.test_system_info()
        self.test_text_generation([
            "What is Python?",
            "Explain Docker in simple terms",
            "Write a hello world program"
        ])
        self.test_embeddings()

        print(f"\n📊 Test Summary:")
        print(f"Passed: {self.success_count}")
        print(f"Failed: {self.fail_count}")
        print(f"Total: {self.success_count + self.fail_count}")

        return self.fail_count == 0

if __name__ == "__main__":
    tester = APITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)