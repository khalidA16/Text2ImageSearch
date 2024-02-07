from qdrant_client import QdrantClient, models
from config import COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY
from utils import get_image_paths, load_embeddings


class UploadQdrant:
    def __init__(self) -> None:
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    def upload(self, image_paths):
        self.image_paths = image_paths
        self.create_collection()
        self.upload_collection()

    def create_collection(self):
        print(f'Creating collection "{COLLECTION_NAME}" from Image Embeddings')
        self.image_embeddings = load_embeddings()

        self.client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=self.image_embeddings[0].size(1), distance=models.Distance.COSINE
            ),
            # to reduce memory usage
            # See: https://github.com/qdrant/qdrant_demo/blob/0c14790d89ab9d2b865aa2832341ab29fd56bb82/qdrant_demo/init_collection_startups.py#L35
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            ),
        )
        print(f'Collection "{COLLECTION_NAME}" created')

    def upload_collection(self):
        print(f'Uploading Collection "{COLLECTION_NAME}" to Qdrant')
        for idx, (embedding, path) in enumerate(
            zip(self.image_embeddings, self.image_paths)
        ):
            point = models.PointStruct(
                id=idx,  # Use loop idx as ID
                vector=embedding.squeeze().tolist(),
                payload={"path": path},
            )
            response = self.client.upsert(
                collection_name=COLLECTION_NAME, points=[point], wait=True
            )
        if "completed" in response.status:
            print(f"Upload Successful!")


if __name__ == "__main__":
    image_paths = get_image_paths()
    qdclient = UploadQdrant()
    qdclient.upload(image_paths)
