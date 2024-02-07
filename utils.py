import pickle
import os
from config import DATASET_PATH
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
        "--load", type=bool, default="True", help="load pre-creted image embeddings"
    )
    args = parser.parse_args()

    return args
