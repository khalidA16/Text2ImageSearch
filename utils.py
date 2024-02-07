import pickle
import os
from config import DATASET_PATH
import argparse


def get_image_paths():
    image_paths = list()
    for file in os.listdir(DATASET_PATH):
        image_paths.append(os.path.join(DATASET_PATH, file))
    return image_paths


def save_embeddings(embeddings, mode):
    file = open(f"{mode}_embeddings", "wb")
    pickle.dump(embeddings, file)
    file.close()


def load_embeddings(mode):
    file = open(f"{mode}_embeddings", "rb")
    embeddings = pickle.load(file)
    file.close()
    return embeddings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=bool, default="True", help="load pre-creted image embeddings")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for ImageEmbedder")
    parser.add_argument('--hit_limit', type=int, default=5, help="Limits of images to show w.r.t the query")
    
    args = parser.parse_args()

    return args