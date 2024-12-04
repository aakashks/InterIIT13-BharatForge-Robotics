import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

db_client = chromadb.PersistentClient('./.chromadb')
embedding_function = OpenCLIPEmbeddingFunction('ViT-B-16-SigLIP', 'webli', device='cuda')

def query_images(query_text, n_results=20, collection_name='embeddings'):
    collection = db_client.get_or_create_collection(name=collection_name, embedding_function=embedding_function, data_loader=ImageLoader())
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )
    
    return results



def send_coords():
    user_input = input("Enter where you wish to go: ")
    
    