import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import asyncio
import aiohttp
import pandas as pd
import json
from tqdm.asyncio import tqdm_asyncio
import os
import gc

# Configuration
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://0.0.0.0:8000/v1"
MODEL_NAME = "allenai/Molmo-7B-D-0924"
CONCURRENT_REQUESTS = 10  # Number of concurrent API requests
BATCH_SIZE = 10  # Number of images to process in a batch
OUTPUT_FOLDER = "./output"  # Folder to save the output JSON

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize the embedding function
embedding_function = OpenCLIPEmbeddingFunction('ViT-B-16-SigLIP', 'webli')  # device='cuda' for GPU

# Initialize the ChromaDB client
client = chromadb.PersistentClient('./db')

# Image storage function
def store_images(data, collection_name='embeddings'):    
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function, data_loader=ImageLoader())
    collection.add(
        ids=data.keys(),        # list of x_y_yaw strings
        uris=[node['image_path'] for node in data],
        metadatas=[data[node] for node in data]
    )

# Image update function
def update_images(new_data, collection_name='embeddings'):
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function, data_loader=ImageLoader())
    for node in new_data:
        collection.update(
            ids=node,
            uri=new_data[node]['image_path'],
            metadata=new_data[node]
        )

# Query function for images
def query_images(query_text, n_results=20, collection_name='embeddings'):
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function, data_loader=ImageLoader())
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )
    return results

# Asynchronous fetch function to send image URLs to the API
async def fetch(session, semaphore, image_url, index, entity_name):
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
                        {"type": "text", "text": "Describe the contents of the image."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
            }

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            request_timeout = aiohttp.ClientTimeout(total=5)  # 5-second timeout

            async with session.post(f"{OPENAI_API_BASE}/chat/completions", json=payload, headers=headers, timeout=request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    vlm_output = data['choices'][0]['message']['content']
                else:
                    vlm_output = f"Error: {response.status}"
        except asyncio.TimeoutError:
            vlm_output = "Timeout Error: Request took longer"
        except Exception as e:
            vlm_output = f"Exception: {str(e)}"

        return {
            'index': int(index),
            'entity_name': entity_name,
            'vlm_output': vlm_output
        }

# Process images in batches asynchronously
async def process_images(session, semaphore, images, start_idx, df_test):
    tasks = []
    for i, image_url in enumerate(images):
        idx = start_idx + i
        entity_name = df_test.at[idx, 'entity_name']
        tasks.append(fetch(session, semaphore, image_url, df_test.at[idx, 'index'], entity_name))
    
    results = await asyncio.gather(*tasks)
    return results

# Main asynchronous function to process all images
async def main_async(df_test, image_paths):
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=None)  # Adjust timeout as needed

    cumulative_results = []
    batch_counter = 1

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for start in tqdm_asyncio(range(0, len(image_paths), BATCH_SIZE), desc="Processing Batches"):
            end = min(start + BATCH_SIZE, len(image_paths))
            batch_images = image_paths[start:end]
            batch_results = await process_images(session, semaphore, batch_images, start, df_test)

            cumulative_results.extend(batch_results)

            # Write to file
            output_path = os.path.join(OUTPUT_FOLDER, f'batch_output_{batch_counter}.json')
            with open(output_path, 'w') as outfile:
                json.dump(batch_results, outfile, indent=4)
            
            print(f"----- Batch {batch_counter} saved with {len(batch_results)} results. -----")
            batch_counter += 1

            # Clear memory
            cumulative_results = []
            gc.collect()

    print("All batches processed successfully.")

# Function to handle text queries
def ask_text_query(text_prompt):
    # For 1 text query, return the output of the VLM
    # Send the prompt to the API and get the results
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": text_prompt}],
        }],
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async def fetch_text_query():
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{OPENAI_API_BASE}/chat/completions", json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    return f"Error: {response.status}"

    return asyncio.run(fetch_text_query())

# Function to handle image queries
def ask_image_query(images_path, prompt):
    # For a list of images, run VLM on each image and return the output
    return ask_text_query(prompt)  # Simplified for now. Integrate image-specific handling later.

def main(user_query, df_test, image_paths):
    # Query to find possible objects from the text input
    prompt1 = f"""
    Given this user query: "{user_query}", list possible names of objects or entities that match closely with the query. Return the names in a JSON format.
    """

    # Get possible objects
    possible_objects = ask_text_query(prompt1)

    # Assuming the output format from the VLM is a JSON array of possible objects
    possible_objects = json.loads(possible_objects).get('possible_objects', [])

    # Create queries for images related to each possible object
    possible_objects_queries = [f'a photo of {obj}' for obj in possible_objects]

    # Query the image database
    results = []
    for query in possible_objects_queries:
        query_result = query_images(query, n_results=5)
        results.append(query_result)

    # Now generate the image query results
    prompt2 = f"""
    Given this user query: "{user_query}", find the most relevant object or entity in the image given. Point to the object and give the appropriate name and description of the image.
    """
    image_query_results = ask_image_query(image_paths, prompt2)

    return {
        "possible_objects": possible_objects,
        "image_results": results,
        "image_query_results": image_query_results
    }
