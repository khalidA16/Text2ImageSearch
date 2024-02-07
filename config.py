import os 

DATASET_PATH = os.path.join(os.getcwd(), "image_dataset/")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "ad_images")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333/")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")