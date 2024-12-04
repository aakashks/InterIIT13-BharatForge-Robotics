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


# def ask_image_query(images_path, prompt):
#     # for a list of images, run vlm on each image and return the output
#     # batch of images
#     # will usually be a small number so send 10 requests at a time
    
    
#     return vlm_output
    


# def main(user_query, data):
    
#     prompt1 = """
#     Given this user query: "{}", list possible names of objects or entities that match closely with the query. Return the names in a json format.
#     """.format(user_query) + "\n" +   """
#     {
#         possible_objects: [
#             "object1",
#             "object2",
#             "object3"
#         ]
#     }
#     """
    
    
#     # run ask_text_query
#     # get a list possible_objects
    
#     possible_objects_queries = [f'a photo of {possile_object}' for possile_object in possible_objects]
    
#     results = query_images
    
    
#     prompt2 = """
#     Given this user query: "{}", find the most relevant object or entity in the image given. Point to the object and give the appropriate name and description of the image.
#     """.format(user_query)
    
#     # run ask_image_query
    
#     return results
    