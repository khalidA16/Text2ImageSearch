from embedder import ImageEmbedder, TextEmbedder
from qdrant_client_upload import UploadQdrant
from utils import get_image_paths, parse_args
import subprocess
from config import COLLECTION_NAME


def get_hit_scores(query: str, embedder, qdclient, limit: int) -> dict:
    query_embedding = embedder([query])
    search_result = qdclient.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.squeeze().tolist(),
        limit=11106,
    )

    all_scores = dict()
    for hit in search_result:
        all_scores[hit.payload["path"]] = hit.score
    return all_scores


def main() -> None:
    """
    Main function to run the text2image search system.

    Returns:
        None
    """
    args = parse_args()
    image_paths = get_image_paths()

    if not args.load:
        # create new image embeddings
        embedder = ImageEmbedder()
        embedder()

    # create qdrant client
    qdclient = UploadQdrant()
    qdclient.upload(image_paths)

    print("Starting Streamlit app...")
    subprocess.Popen(["streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
