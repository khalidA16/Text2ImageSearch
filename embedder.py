import os
from typing import Any
import torch
from transformers import CLIPModel, CLIPProcessor
from config import DATASET_PATH
from PIL import Image
from torch.utils.data import DataLoader
from collections import defaultdict
import pickle
import torch.multiprocessing as mp
from queue import Queue


class Embedder:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def save_embeddings(self, embeddings, mode):
        file = open(f"{mode}_embeddings", "wb")
        pickle.dump(embeddings, file)
        file.close()

    def load_embeddings(self, mode):
        file = open(f"{mode}_embeddings", "wb")
        embeddings = pickle.load(file)
        file.close()
        return embeddings


class ImageEmbedder(Embedder):

    def __init__(self):
        super().__init__()
        self.mode = "image"

    def __call__(self, images_paths):
        embeddings = list()
        for img_path in images_paths:
            inputs = self.processor(
                images=Image.open(img_path), return_tensors="pt"
            ).to(self.device)
        with torch.no_grad():
            embeddings.append(self.model.get_image_features(**inputs).cpu())
        self.save_embeddings(embeddings)
        return embeddings

        # batch processing of images
        # embeddings = BatchProcessImages()
        # return embeddings

        # Parallel processing of images
        # embeddings = ParallelProcessImages(images_paths)
        # return embeddings


class TextEmbedder(Embedder):

    def __init__(self):
        super().__init__()
        self.mode = "text"

    def __call__(self, text):
        inputs = self.processor(text, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs).cpu()
        return embedding


class BatchProcessImages(ImageEmbedder):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def __call__(self, images_paths):
        # TODO: computation time is same with and without batches
        dataloader = DataLoader(images_paths, batch_size=self.batch_size, shuffle=False)

        image_embeddings = list()

        # Process batches of images
        for batch_paths in dataloader:
            batch_images = [
                self.processor(images=Image.open(img_path), return_tensors="pt")
                for img_path in batch_paths
            ]

            # Use defaultdict to automatically initialize lists for keys
            batch_inputs = defaultdict(list)
            for img_dict in batch_images:
                for key, value in img_dict.items():
                    batch_inputs[key].append(value.to(self.device))

            # Convert lists to tensors
            batch_inputs = {
                key: torch.stack(value_list).squeeze(1)
                for key, value_list in batch_inputs.items()
            }

            with torch.no_grad():
                batch_embeddings = self.model.get_image_features(**batch_inputs)
                image_embeddings.append(batch_embeddings.cpu())

            self.save_embeddings(image_embeddings)
            return image_embeddings


class ParallelProcessImages(ImageEmbedder):
    # Process images in parallel using multiprocessing

    def __call__(self, images_paths, num_workers=4):
        output_queue = Queue()
        processes = []
        for img_path in images_paths:
            process = mp.Process(
                target=self.process_image, args=(img_path, output_queue)
            )
            process.start()
            processes.append(process)

        # Wait for all processes to finish
        for process in processes:
            process.join()

        # Retrieve image embeddings from the output queue
        image_embeddings = []
        while not output_queue.empty():
            image_embeddings.append(output_queue.get())
            self.save_embeddings(image_embeddings.cpu())
        return image_embeddings

    def process_image(self, img_path, output_queue):
        # Define a worker function to process images
        image = Image.open(img_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)
        output_queue.put(image_embedding)


if __name__ == "__main__":
    embd = ImageEmbedder()
    images_paths = list()
    for file in os.listdir(DATASET_PATH):
        images_paths.append(os.path.join(DATASET_PATH, file))
    embd(images_paths)
