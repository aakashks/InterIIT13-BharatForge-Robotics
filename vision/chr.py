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


embedding_function = OpenCLIPEmbeddingFunction('ViT-B-16-SigLIP', 'webli')    # device='cuda' for GPU

# format
# {
#     '12_34_180': {
#         'image_path': '12_34_180.jpg',
#         'label': '12_34_180'
#     },
#     '12_34_270': {
#         'image_path': '12_34_270.jpg',
#         'label': '12_34_270'
#     }
# }

def store_images(data, collection_name='embeddings'):    
    client = chromadb.PersistentClient('./db')
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function, data_loader=ImageLoader())
    collection.add(
        ids=data.keys(),        # list of x_y_yaw strings
        uris=[node['image_path'] for node in data],
        metadatas=[data[node] for node in data]
    )
    
def update_images(new_data, collection_name='embeddings'):
    client = chromadb.PersistentClient('./db')
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function, data_loader=ImageLoader())
    for node in new_data:
        collection.update(
            ids=node,
            uri=new_data[node]['image_path'],
            metadata=new_data[node]
        )
        
def query_images(query_text, n_results=20, collection_name='embeddings'):
    client = chromadb.PersistentClient('./db')
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function, data_loader=ImageLoader())
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )
    
    return results


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
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
            }

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            # Set a specific timeout for the request
            request_timeout = aiohttp.ClientTimeout(total=5)  # 5-second timeout

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

        return {
            'index': int(index),
            'entity_name': entity_name,
            'vlm_output': vlm_output
        }



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
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
            }

            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            # Set a specific timeout for the request
            request_timeout = aiohttp.ClientTimeout(total=5)  # 5-second timeout

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

        return {
            'index': int(index),
            'entity_name': entity_name,
            'vlm_output': vlm_output
        }

async def process_images(session, semaphore, images, start_idx):
    """
    Process a list of images asynchronously.
    """
    tasks = []
    for i, image_url in enumerate(images):
        idx = start_idx + i
        entity_name = df_test.at[idx, 'entity_name']
        tasks.append(fetch(session, semaphore, image_url, df_test.at[idx, 'index'], entity_name))
    
    results = await asyncio.gather(*tasks)
    return results

async def main_async():
    """
    Main asynchronous function to process all images.
    """
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=None)  # Adjust timeout as needed

    cumulative_results = []
    batch_counter = 1

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for start in tqdm_asyncio(range(0, len(image_paths), BATCH_SIZE), desc="Processing Batches"):
            end = min(start + BATCH_SIZE, len(image_paths))
            batch_images = image_paths[start:end]
            batch_results = await process_images(session, semaphore, batch_images, start)

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


def ask_text_query(text_prompt):
    # for 1 text query return the output of the vlm
    # here no image is involved
    
    return vlm_output


def ask_image_query(images_path, prompt):
    # for a list of images, run vlm on each image and return the output
    # batch of images
    # will usually be a small number so send 10 requests at a time
    
    
    return vlm_output
    


def main(user_query, data):
    
    prompt1 = """
    Given this user query: "{}", list possible names of objects or entities that match closely with the query. Return the names in a json format.
    """.format(user_query) + "\n" +   """
    {
        possible_objects: [
            "object1",
            "object2",
            "object3"
        ]
    }
    """
    
    
    # run ask_text_query
    # get a list possible_objects
    
    possible_objects_queries = [f'a photo of {possile_object}' for possile_object in possible_objects]
    
    results = query_images
    
    
    prompt2 = """
    Given this user query: "{}", find the most relevant object or entity in the image given. Point to the object and give the appropriate name and description of the image.
    """.format(user_query)
    
    # run ask_image_query
    
    return results
    