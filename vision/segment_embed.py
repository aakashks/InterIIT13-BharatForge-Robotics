import io

import clip  # CLIP model
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms, box_area
from torchvision.transforms import ToTensor


def build_efficient_sam_vitt(device='cpu'):
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="EfficientSAM/weights/efficient_sam_vitt.pt",
    ).eval().to(device)

def build_efficient_sam_vits(device='cpu'):
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint="EfficientSAM/weights/efficient_sam_vits.pt",
    ).eval().to(device)

def process_small_region(rles):
    # Function as before to remove small regions
    new_masks = []
    scores = []
    min_area = 100
    nms_thresh = 0.7
    for rle in rles:
        mask = rle_to_mask(rle[0])
        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode="islands")
        unchanged = unchanged and not changed

        new_masks.append(torch.as_tensor(mask).unsqueeze(0))
        scores.append(float(unchanged))

    # Recalculate boxes and remove duplicates using NMS
    masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores),
        torch.zeros_like(boxes[:, 0]),
        iou_threshold=nms_thresh,
    )

    # Recalculate RLEs for changed masks
    for i_mask in keep_by_nms:
        if scores[i_mask] == 0.0:
            mask_torch = masks[i_mask].unsqueeze(0)
            rles[i_mask] = mask_to_rle_pytorch(mask_torch)
    masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
    return masks

def get_predictions_given_embeddings_and_queries(img, points, point_labels, model):
    predicted_masks, predicted_iou = model(
        img[None, ...], points, point_labels
    )
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_masks = torch.take_along_dim(
        predicted_masks, sorted_ids[..., None, None], dim=2
    )
    predicted_masks = predicted_masks[0]
    iou = predicted_iou_scores[0, :, 0]
    index_iou = iou > 0.7
    iou_ = iou[index_iou]
    masks = predicted_masks[index_iou]
    score = calculate_stability_score(masks, 0.0, 1.0)
    score = score[:, 0]
    index = score > 0.9
    score_ = score[index]
    masks = masks[index]
    iou_ = iou_[index]
    masks = torch.ge(masks, 0.0)
    return masks, iou_

def show_anns_ours(mask, ax):
    ax.set_autoscale_on(False)
    img = np.ones((mask[0].shape[0], mask[0].shape[1], 4))
    img[:,:,3] = 0
    for ann in mask:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    ax.imshow(img)


def get_embeddings(image_path, grid_size=16, sam_model='t', clip_model='ViT-B/32', device='cpu'):
    clip_model, preprocess = clip.load(clip_model, device=device)
    
    if sam_model == 't':
        model = build_efficient_sam_vitt(device=device)
    else:
        model = build_efficient_sam_vits(device=device)

    model = model.cpu()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image)
    _, original_image_h, original_image_w = img_tensor.shape
    xy = []
    for i in range(grid_size):
        curr_x = 0.5 + i / grid_size * original_image_w
        for j in range(grid_size):
            curr_y = 0.5 + j / grid_size * original_image_h
            xy.append([curr_x, curr_y])
    xy = torch.from_numpy(np.array(xy))
    points = xy
    num_pts = xy.shape[0]
    point_labels = torch.ones(num_pts, 1)
    with torch.no_grad():
        predicted_masks, predicted_iou = get_predictions_given_embeddings_and_queries(
              img_tensor.cpu(),
              points.reshape(1, num_pts, 1, 2).cpu(),
              point_labels.reshape(1, num_pts, 1).cpu(),
              model.cpu(),
          )
    rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
    predicted_masks = process_small_region(rle)

    # Use CLIP to generate embeddings for each mask
    mask_embeddings = []
    for mask in predicted_masks:
        mask = Image.fromarray(mask)
        mask = preprocess(mask).unsqueeze(0).to(device)
        with torch.no_grad():
            mask_embedding = clip_model.encode_image(mask)
        mask_embeddings.append(mask_embedding)
        
    # return predicted_masks, mask_embeddings
    return mask_embeddings



def store_embeddings(image_id, segment_id, image_path, database_path, **kwargs):
    embeddings = get_embeddings(image_path, **kwargs)
    qdrant_client = QdrantClient(':memory:')
    collection_name = "image_embeddings"
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vector_size=512,  # Dimension of the embedding vector
        distance="Cosine"  # Metric for similarity
    )
    
    points = [
        PointStruct(
            id=f"{image_id}_{segment_id}",  # Unique ID for each point
            vector=embedding.tolist(),     # Convert numpy array to list for JSON serialization
            payload={"image_id": image_id, "segment_id": segment_id}  # Metadata
        )
        for image_id, segment_id, embedding in embeddings
    ]

    qdrant_client.upsert(collection_name=collection_name, points=points)
    
    
def query_embeddings(image_path, database_path, **kwargs):
    query_embeddings = get_embeddings(image_path, **kwargs)
    
    qdrant_client = QdrantClient(database_path)
    
    collection_name = "image_embeddings"
    
    results = qdrant_client.search(
        collection_name=collection_name,
        vectors=[embedding.tolist() for embedding in query_embeddings],
        limit=5  # Retrieve top 5 closest embeddings
    )
        
    for result in results:
        print(f"Match ID: {result.id}, Distance: {result.score}, Metadata: {result.payload}")