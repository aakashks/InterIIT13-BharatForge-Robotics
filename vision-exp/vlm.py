import asyncio
import aiohttp
import pandas as pd
import json
from tqdm.asyncio import tqdm_asyncio
import os
import gc
import sys

# Configuration
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://0.0.0.0:8000/v1"
MODEL_NAME = "allenai/Molmo-7B-D-0924"
CONCURRENT_REQUESTS = 20  # Number of concurrent API requests
TIMEOUT = 10  # Timeout in seconds

async def fetch(session, semaphore, prompt, image_url):
    """
    Asynchronously fetch the API response for a single image.
    """
    async with semaphore:
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": 'file://' + image_url}},
                    ],
                }],
            }

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            # Set a specific timeout for the request
            request_timeout = aiohttp.ClientTimeout(total=TIMEOUT)  # 5-second timeout

            async with session.post(f"{OPENAI_API_BASE}/chat/completions", json=payload, headers=headers, timeout=request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    # Adjust based on actual response structure
                    vlm_output = data['choices'][0]['message']['content']
                else:
                    vlm_output = f"Error: {response.status}"
        except asyncio.TimeoutError:
            vlm_output = "Timeout Error: Request took longer"
        except Exception as e:
            vlm_output = f"Exception: {str(e)}"

        return vlm_output


async def run_multiple_image_query(image_dir, prompt):
    """
    Main asynchronous function to process all images.
    """
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    # get absolute path of images
    image_paths = [os.path.abspath(f) for f in image_paths]
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=None)  # Adjust timeout as needed

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        tasks = []
        for image_url in image_paths:
            tasks.append(fetch(session, semaphore, prompt, image_url))
        
        results = await asyncio.gather(*tasks)

        gc.collect()
        
    with open("results.json", "w") as f:
        json.dump(results, f)
        
    return results

if __name__ == "__main__":
    image_dir = sys.argv[1]
    prompt = sys.argv[2]
    asyncio.run(run_multiple_image_query(image_dir, prompt))
    