"""
Performance testing script for TalentJobMatch API
This script will help identify bottlenecks in the /api/v1/match/candidate endpoint
"""

import time
import requests
import json
from typing import Dict, Any
import sys

def test_endpoint_performance(url: str, payload: Dict[str, Any]):
    """
    Test the performance of the match candidate endpoint
    """
    print(f"Testing endpoint: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        response = requests.post(
            url, 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Total Response Time: {total_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Number of Matches: {result.get('total_candidates', 0)}")
            print(f"Top Matches: {len(result.get('top_matches', []))}")
            
            if result.get('top_matches'):
                print("\nTop 3 Matches:")
                for i, match in enumerate(result['top_matches'][:3], 1):
                    print(f"  {i}. {match.get('name', 'N/A')} - Score: {match.get('score', 0):.3f}")
        else:
            print(f"Error Response: {response.text}")
            
        return total_time, response.status_code
        
    except requests.exceptions.Timeout:
        print("Request timed out!")
        return None, "TIMEOUT"
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return None, "ERROR"

def main():
    # Test configuration
    base_url = "http://127.0.0.1:8000"
    endpoint = "/api/v1/match/candidate"
    full_url = base_url + endpoint
    
    # Sample payloads for testing
    test_payloads = [
        {
            "description": "Looking for a Frontend Engineer skilled in React with 3 years experience building scalable web applications."
        },
        {
            "description": "Seeking a Python developer with experience in Django, Flask, and REST APIs for backend development."
        },
        {
            "description": "Need a Full Stack Developer with expertise in JavaScript, Node.js, React, and MongoDB."
        }
    ]
    
    print("🚀 Starting Performance Tests for TalentJobMatch API")
    print("=" * 60)
    
    results = []
    
    for i, payload in enumerate(test_payloads, 1):
        print(f"\n📋 Test {i}/{len(test_payloads)}")
        print("-" * 40)
        
        time_taken, status = test_endpoint_performance(full_url, payload)
        results.append({
            "test_num": i,
            "payload": payload,
            "time_taken": time_taken,
            "status": status
        })
        
        if time_taken and time_taken > 30:  # If it takes more than 30 seconds
            print(f"⚠️  Warning: Test {i} took {time_taken:.2f} seconds, which is quite long.")
        
        # Add delay between tests to avoid overwhelming the server
        if i < len(test_payloads):
            print("\n⏳ Waiting 5 seconds before next test...\n")
            time.sleep(5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    successful_tests = [r for r in results if r['time_taken'] is not None and r['status'] == 200]
    
    if successful_tests:
        avg_time = sum(r['time_taken'] for r in successful_tests) / len(successful_tests)
        max_time = max(r['time_taken'] for r in successful_tests)
        min_time = min(r['time_taken'] for r in successful_tests)
        
        print(f"Successful Tests: {len(successful_tests)}/{len(results)}")
        print(f"Average Response Time: {avg_time:.2f} seconds")
        print(f"Min Response Time: {min_time:.2f} seconds")
        print(f"Max Response Time: {max_time:.2f} seconds")
        
        if avg_time > 30:
            print("\n🔴 CRITICAL: Average response time is very high (>30s)")
            print("Consider the following optimizations:")
            print("  1. Optimize vector database queries")
            print("  2. Reduce number of LLM calls")
            print("  3. Implement caching for expensive operations")
            print("  4. Use smaller embedding models")
        elif avg_time > 15:
            print("\n🟡 WARNING: Average response time is high (>15s)")
            print("Consider optimization strategies")
        else:
            print("\n🟢 GOOD: Average response time is reasonable")
    else:
        print("❌ No successful tests completed")
    
    print("\n📋 Detailed Results:")
    for result in results:
        status_icon = "✅" if result['status'] == 200 else "❌"
        time_str = f"{result['time_taken']:.2f}s" if result['time_taken'] else "Failed"
        print(f"  {status_icon} Test {result['test_num']}: {time_str} - {result['status']}")

if __name__ == "__main__":
    main()