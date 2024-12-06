import os
import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms, box_area
from torchvision.transforms import ToTensor


def divide_image(image, l=4, w=3):
    # Divide the image into l x w grid
    h, w, _ = image.shape
    l, w = l, w
    l_size, w_size = h // l, w // w
    images = []
    for i in range(l):
        for j in range(w):
            images.append(image[i*l_size:(i+1)*l_size, j*w_size:(j+1)*w_size])
    return images


def get_embeddings_batch(list_images, model, preprocess, device='cpu'):
    model = model.eval().to(device)
    
    images = torch.stack([preprocess(img) for img in list_images]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(images)
        
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy()
    
    


def update_db():
    pass



def store_embeddings(list_images, database_path, collection_name):
    qdrant_client = QdrantClient(path=database_path)
    
    qdrant_client.create_collection(            # use recreate to overwrite existing collection
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=512,
            distance=Distance.DOT,
        )
    )

    points = [
        PointStruct(
            id=image_id * 1000 + segment_id,
            vector=embedding.tolist(),
            payload={"image_id": image_id, "image_path": image_path, "segment_id": segment_id}
        )
        for segment_id, embedding in embeddings.items()
    ]

    op = qdrant_client.upsert(collection_name=collection_name, points=points, wait=True)
    print(op)
    
    
def query_embeddings(text_query, clip_model, database_path, collection_name, topk=10, device='cpu'):
    clip_model = clip_model.to(device)
    text_tokens = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).cpu().squeeze(0)
    
    print(text_features.shape)
    print(text_features.tolist())
    
    qdrant_client = QdrantClient(database_path)
    
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=text_features.tolist(),
        with_payload=True,
        limit=topk
    )
        
    for result in results:
        print(f"Match ID: {result.id}, Distance: {result.score}, Metadata: {result.payload}")
        
        
if __name__ == "__main__":
    database_path = "./db"
    collection_name = "small_images"
    
    model, clip_model, clip_preprocess = load_models()
    [store_embeddings(1, img, database_path, collection_name=collection_name, model=model, 
                     clip_model=clip_model, clip_preprocess=clip_preprocess, device='cuda') for img in os.listdir('../images')]
    
    # query_embeddings("fire extinguisher", clip_model, database_path, collection_name, device='cuda')
