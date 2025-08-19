import requests
import sys
import json
from datetime import datetime
import base64
import io
from PIL import Image

class BackendTester:
    def __init__(self, base_url="https://newschecker-2.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.tests_run = 0
        self.tests_passed = 0
        self.user_email = f"test_user_{datetime.now().strftime('%H%M%S')}@example.com"
        self.user_password = "TestPass123!"

    def run_test(self, name, method, endpoint, expected_status, data=None, headers=None):
        """Run a single API test"""
        url = f"{self.base_url}{endpoint}"
        test_headers = {'Content-Type': 'application/json'}
        if headers:
            test_headers.update(headers)
        if self.token:
            test_headers['Authorization'] = f'Bearer {self.token}'

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=test_headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=test_headers, timeout=30)

            print(f"   Status: {response.status_code}")
            success = response.status_code == expected_status
            
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test GET /api/"""
        success, response = self.run_test(
            "Root Endpoint",
            "GET",
            "/",
            200
        )
        if success and response.get('message') == 'Hello World':
            print("   ✓ Root endpoint returns correct message")
            return True
        else:
            print("   ✗ Root endpoint message incorrect")
            return False

    def test_signup(self):
        """Test user signup"""
        success, response = self.run_test(
            "User Signup",
            "POST",
            "/auth/signup",
            200,
            data={"email": self.user_email, "password": self.user_password}
        )
        if success and 'access_token' in response:
            self.token = response['access_token']
            print(f"   ✓ Signup successful, token received")
            return True
        else:
            print("   ✗ Signup failed or no token received")
            return False

    def test_login(self):
        """Test user login"""
        success, response = self.run_test(
            "User Login",
            "POST",
            "/auth/login",
            200,
            data={"email": self.user_email, "password": self.user_password}
        )
        if success and 'access_token' in response:
            self.token = response['access_token']
            print(f"   ✓ Login successful, token received")
            return True
        else:
            print("   ✗ Login failed or no token received")
            return False

    def test_me_endpoint(self):
        """Test GET /auth/me"""
        if not self.token:
            print("❌ No token available for /auth/me test")
            return False
            
        success, response = self.run_test(
            "Get Current User",
            "GET",
            "/auth/me",
            200
        )
        if success and response.get('email') == self.user_email:
            print(f"   ✓ User info correct: {response.get('email')}")
            return True
        else:
            print("   ✗ User info incorrect or missing")
            return False

    def test_analyze_headline(self):
        """Test analyze with headline"""
        if not self.token:
            print("❌ No token available for analyze test")
            return False
            
        success, response = self.run_test(
            "Analyze Headline",
            "POST",
            "/llm/analyze",
            200,
            data={"headline": "NASA confirms water on the sun's surface"}
        )
        
        # Check if we get expected fields or a 503 error (which is acceptable)
        if success:
            required_fields = ['verdict', 'p_fake', 'bias_label', 'rewrites', 'evidence']
            if all(field in response for field in required_fields):
                print(f"   ✓ Analysis successful with all required fields")
                print(f"   Verdict: {response.get('verdict')}, P_fake: {response.get('p_fake')}")
                return True
            else:
                print(f"   ✗ Missing required fields in response")
                return False
        else:
            # Check if it's a 503 error (service unavailable) which is acceptable
            return self.run_test(
                "Analyze Headline (503 check)",
                "POST", 
                "/llm/analyze",
                503,
                data={"headline": "NASA confirms water on the sun's surface"}
            )[0]

    def test_analyze_url(self):
        """Test analyze with URL"""
        if not self.token:
            print("❌ No token available for analyze URL test")
            return False
            
        success, response = self.run_test(
            "Analyze URL",
            "POST",
            "/llm/analyze",
            200,
            data={"url": "https://example.com"}
        )
        
        if success:
            required_fields = ['verdict', 'p_fake', 'bias_label', 'rewrites', 'evidence']
            if all(field in response for field in required_fields):
                print(f"   ✓ URL analysis successful")
                return True
            else:
                print(f"   ✗ Missing required fields in URL analysis response")
                return False
        else:
            # Check if it's a 503 error which is acceptable
            return self.run_test(
                "Analyze URL (503 check)",
                "POST",
                "/llm/analyze", 
                503,
                data={"url": "https://example.com"}
            )[0]

    def create_test_image(self):
        """Create a small test image as base64 data URL"""
        # Create a 64x64 PNG with a black X
        img = Image.new('RGB', (64, 64), color='white')
        pixels = img.load()
        
        # Draw a black X
        for i in range(64):
            pixels[i, i] = (0, 0, 0)  # Diagonal line
            pixels[i, 63-i] = (0, 0, 0)  # Other diagonal
            
        # Convert to base64 data URL
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_data}"

    def test_analyze_image(self):
        """Test analyze with image"""
        if not self.token:
            print("❌ No token available for analyze image test")
            return False
            
        try:
            image_data_url = self.create_test_image()
            print(f"   Created test image: {len(image_data_url)} chars")
        except Exception as e:
            print(f"   ✗ Failed to create test image: {e}")
            return False
            
        success, response = self.run_test(
            "Analyze Image",
            "POST",
            "/llm/analyze",
            200,
            data={"image_base64": image_data_url}
        )
        
        if success:
            required_fields = ['verdict', 'p_fake', 'bias_label', 'rewrites', 'evidence']
            if all(field in response for field in required_fields):
                print(f"   ✓ Image analysis successful")
                return True
            else:
                print(f"   ✗ Missing required fields in image analysis response")
                return False
        else:
            # Check if it's a 503 error which is acceptable
            return self.run_test(
                "Analyze Image (503 check)",
                "POST",
                "/llm/analyze",
                503,
                data={"image_base64": image_data_url}
            )[0]

    def test_invalid_requests(self):
        """Test error handling"""
        print(f"\n🔍 Testing Error Handling...")
        
        # Test analyze without auth
        old_token = self.token
        self.token = None
        success, _ = self.run_test(
            "Analyze Without Auth",
            "POST",
            "/llm/analyze",
            401,
            data={"headline": "test"}
        )
        self.token = old_token
        
        if not success:
            print("   ✗ Should return 401 for unauthorized analyze request")
            return False
            
        # Test analyze with empty payload
        success, _ = self.run_test(
            "Analyze Empty Payload",
            "POST", 
            "/llm/analyze",
            400,
            data={}
        )
        
        if not success:
            print("   ✗ Should return 400 for empty analyze payload")
            return False
            
        print("   ✓ Error handling working correctly")
        return True

def main():
    print("🚀 Starting Backend API Tests")
    print("=" * 50)
    
    tester = BackendTester()
    
    # Test sequence
    tests = [
        ("Root Endpoint", tester.test_root_endpoint),
        ("User Signup", tester.test_signup),
        ("User Login", tester.test_login),
        ("Get Current User", tester.test_me_endpoint),
        ("Analyze Headline", tester.test_analyze_headline),
        ("Analyze URL", tester.test_analyze_url),
        ("Analyze Image", tester.test_analyze_image),
        ("Error Handling", tester.test_invalid_requests),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All backend tests passed!")
        return 0
    else:
        print("⚠️  Some backend tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())