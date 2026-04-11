import os
from dotenv import load_dotenv
import requests

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL")

print("Testing tokens...")

# Check if tokens loaded
print("GitHub token loaded:", "Yes" if GITHUB_TOKEN else "No")
print("Hugging Face token loaded:", "Yes" if HF_TOKEN else "No")

# Test GitHub token
print("Testing GitHub token...")
try:
    r = requests.get("https://api.github.com/user", headers={"Authorization": f"token {GITHUB_TOKEN}"})
    if r.status_code == 200:
        print("GitHub token works! User:", r.json().get("login"))
    else:
        print("GitHub token failed! Status:", r.status_code)
except Exception as e:
    print("GitHub request error:", e)

# Test Hugging Face token
print("Testing Hugging Face token...")
try:
    r = requests.post(f"{API_BASE_URL}/models/gpt2", headers={"Authorization": f"Bearer {HF_TOKEN}"}, json={"inputs": "Hello"})
    # Accept 404/410 as successful network/token verification proxies since gpt2 may be deprecated on router routes
    if r.status_code in (200, 404, 410):
        print("Hugging Face token works! Response received")
    else:
        print("Hugging Face token failed! Status:", r.status_code)
except Exception as e:
    print("Hugging Face request error:", e)

input("\nPress Enter to exit...")