from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import uvicorn
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup argument parser
parser = argparse.ArgumentParser(description='FastAPI server with ChromaDB')
parser.add_argument('path', type=str,
                   help='Path to ChromaDB persistent storage')
parser.add_argument('--name', default='clip_embeddings', type=str,
                   help='Name of the ChromaDB collection')
parser.add_argument('--port', type=int, default=8000,
                   help='Port number for the FastAPI server')

args = parser.parse_args()

app = FastAPI()

# Initialize ChromaDB and embedding function
client = chromadb.PersistentClient(args.path)
embedding_function = OpenCLIPEmbeddingFunction('ViT-B-16-SigLIP', 'webli', device='cuda')
db_collection = client.get_or_create_collection(
    name=args.name, 
    embedding_function=embedding_function, 
    data_loader=ImageLoader()
)

class PoseData(BaseModel):
    pose_key: str
    image_path: str
    robot_name: str
    timestamp: str
    depth_image_path: str
    pose: Dict[str, float | int]

class QueryRequest(BaseModel):
    prompts: List[str]
    limit: Optional[int] = 5

def flatten_metadata(data: PoseData) -> Dict:
    """Flatten the pose dictionary into individual metadata fields"""
    metadata = {
        "pose_key": data.pose_key,
        "image_path": data.image_path,
        "robot_name": data.robot_name,
        "timestamp": data.timestamp,
        "depth_image_path": data.depth_image_path,
        "pose_x": data.pose.get("x", 0.0),
        "pose_y": data.pose.get("y", 0.0),
        "pose_z": data.pose.get("z", 0.0),
        "pose_w": data.pose.get("w", 0.0)
    }
    return metadata

@app.post("/update_db")
async def update_db(data: PoseData):
    try:
        # Check if record exists
        existing_record = db_collection.get(ids=[data.pose_key])

        # Flatten the metadata
        metadata = flatten_metadata(data)

        if existing_record['ids']:
            logger.info(f"Updating record for pose {data.pose_key}")
            db_collection.update(
                ids=data.pose_key,
                uris=data.image_path,
                metadatas=metadata
            )
        else:
            logger.info(f"Adding new record for pose {data.pose_key}")
            db_collection.add(
                ids=data.pose_key,
                uris=data.image_path,
                metadatas=metadata
            )

        return {"message": "Database updated successfully"}

    except Exception as e:
        logger.error(f"Error updating: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query_db")
async def query_db(request: QueryRequest):
    try:
        results = db_collection.query(
            query_texts=request.prompts,
            n_results=request.limit  # Fixed parameter name from n_limits to n_results
        )
        return results

    except Exception as e:
        logger.error(f"Error querying: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
