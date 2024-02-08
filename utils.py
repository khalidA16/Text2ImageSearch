import pickle
import os
from config import DATASET_PATH, COLLECTION_NAME
import argparse


def get_image_paths() -> None:
    """
    Retrieves paths of all images in the dataset directory.

    Returns:
        list: A list containing paths of all images in the dataset directory.
    """
    image_paths = list()
    for file in os.listdir(DATASET_PATH):
        image_paths.append(os.path.join(DATASET_PATH, file))
    return image_paths


def save_embeddings(embeddings: list) -> None:
    """
    Saves image embeddings to a binary file.

    Args:
        embeddings (list): A list containing image embeddings to be saved.

    Returns:
        None
    """

    with open("image_embeddings", "wb") as file:
        pickle.dump(embeddings, file)


def load_embeddings() -> list:
    """
    Loads image embeddings from a binary file.

    Returns:
        list: A list containing the loaded image embeddings.
    """
    with open("image_embeddings", "rb") as file:
        embeddings = pickle.load(file)
    return embeddings


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        Namespace: An object containing parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load", action="store_true", help="load saved image embeddings"
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=32,
        help="Batch size for ImageEmbedder to create image embeddings",
    )
    args = parser.parse_args()

    return args

# used for evaluation only
def get_hit_scores(query: str, embedder, qdclient) -> dict:
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
