import requests
import json

# Define the API server URL
API_BASE_URL = "http://127.0.0.1:5000/api"

def test_search_documents():
    """
    Test the /api/search endpoint.
    """
    url = f"{API_BASE_URL}/search"
    payload = {
        "query": "What are the key environmental factors in sustainable procurement?",
        "dataset": "education",  # Change to "health" to test the health dataset
        "top_n": 3
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Search Documents Response:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"An error occurred: {e}")

def test_generate_text():
    """
    Test the /api/generate endpoint.
    """
    url = f"{API_BASE_URL}/generate"
    payload = {
        "query": "What are the key environmental factors in sustainable procurement?",
        "context": "Sustainable procurement focuses on reducing environmental harm, including carbon footprint, energy efficiency, and resource circularity."
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Generate Text Response:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Error: {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("Testing /api/search endpoint...")
    test_search_documents()
    print("\nTesting /api/generate endpoint...")
    test_generate_text()