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
import subprocess
from torchvision.ops.boxes import batched_nms, box_area
from torchvision.transforms import ToTensor


def build_efficient_sam_vitt():
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="EfficientSAM/weights/efficient_sam_vitt.pt",
    ).eval()

def build_efficient_sam_vits():
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint="EfficientSAM/weights/efficient_sam_vits.pt",
    ).eval()

def load_models(sam='vitt', name='ViT-B/16', device='cpu'):
    model = build_efficient_sam_vitt() if sam == 'vitt' else build_efficient_sam_vits()
    model = model.to(device)
    clip_model, clip_preprocess = clip.load(name, device=device)
    return model, clip_model, clip_preprocess

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

def get_predictions_given_embeddings_and_queries(img, points, point_labels, model, device='cpu'):
    img = img.to(device)
    points = points.to(device)
    point_labels = point_labels.to(device)
    model = model.eval().to(device)
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


def get_embeddings(image_path, model, clip_model, clip_preprocess, grid_size=8, device='cpu'):
    model = model.eval()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image).to(device)
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
              img_tensor,
              points.reshape(1, num_pts, 1, 2).to(device),
              point_labels.reshape(1, num_pts, 1).to(device),
              model,
              device=device,
          )
    rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
    predicted_masks = process_small_region(rle)
    
    print(f"Number of segments: {len(predicted_masks)}")

    # Use CLIP to generate embeddings for each segment
    clip_model = clip_model.to(device)
    embeddings = {}
    iou_scores = {}
    for idx, mask in enumerate(predicted_masks):
        masked_image = image * mask[:, :, None]
        masked_image_pil = Image.fromarray(masked_image)
        
        input_tensor = clip_preprocess(masked_image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(input_tensor).cpu()
        
        image_features /= image_features.norm(dim=-1, keepdim=True).numpy()
        embeddings[idx] = image_features
        iou_scores[idx] = predicted_iou[idx].item()
        

    print('Embeddings generated')
        
    return embeddings, iou_scores



def store_embeddings(image_id, image_path, database_path, collection_name="image_embeddings", **kwargs):
    embeddings, _ = get_embeddings(image_path, **kwargs)
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
    # pip install git+https://github.com/facebookresearch/segment-anything.git
    # make sure you are in a separate new directory
    subprocess.run("git clone https://github.com/yformer/EfficientSAM.git", shell=True, check=True)
    
    # image_path = "image.jpg"
    database_path = "./db"
    collection_name = "example2"
    
    model, clip_model, clip_preprocess = load_models()
    # store_embeddings(1, image_path, database_path, collection_name=collection_name, model=model, clip_model=clip_model, clip_preprocess=clip_preprocess, device='cuda')
    query_embeddings("fire extinguisher", clip_model, database_path, collection_name, device='cuda')
