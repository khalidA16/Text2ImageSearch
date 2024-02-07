from typing import Any
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.multiprocessing as mp
from queue import Queue
from utils import get_image_paths, save_embeddings


class Embedder:
    """
    Base class for embedding text or images using the CLIP model.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class TextEmbedder(Embedder):
    """
    Class for embedding text using the CLIP model.
    """

    def __init__(self):
        super().__init__()
        self.mode = "text"

    def __call__(self, text: str) -> torch.Tensor:
        """
        Embeds text using the CLIP model.

        Args:
            text (str): The text to be embedded.

        Returns:
            torch.Tensor: The embedded text.
        """
        inputs = self.processor(text, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            embedding = self.model.get_text_features(**inputs).cpu()
        return embedding


class ImageEmbedder(Embedder):
    """
    Class for embedding images using the CLIP model.
    """

    def __init__(self):
        super().__init__()
        self.mode = "image"

    def __call__(self, image_paths: list) -> None:
        """
        Embeds images using the CLIP model.

        Args:
            image_paths (list): A list of paths to the images.

        Returns:
            None
        """
        embeddings = list()
        for img_path in image_paths:
            inputs = self.processor(
                images=Image.open(img_path), return_tensors="pt"
            ).to(self.device)
        with torch.no_grad():
            embeddings.append(self.model.get_image_features(**inputs).cpu())
        save_embeddings(embeddings)

        # batch processing of images
        # embeddings = BatchProcessImages(images_paths)
        # return embeddings

        # Parallel processing of images
        # embeddings = ParallelProcessImages(images_paths)
        # return embeddings


class BatchProcessImages(ImageEmbedder):
    """
    Class for batch processing images using the CLIP model.
    """

    def __init__(self, batch_size=32) -> None:
        """
        Initializes the BatchProcessImages class.

        Args:
            batch_size (int): The batch size for processing images.
        """
        super().__init__()
        self.batch_size = batch_size

    def __call__(self, image_paths) -> None:
        """
        Batch processes images using the CLIP model.

        Args:
            image_paths (list): A list of paths to the images.

        Returns:
            None
        """
        dataloader = DataLoader(image_paths, batch_size=self.batch_size, shuffle=False)

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

            save_embeddings(image_embeddings)


class ParallelProcessImages(ImageEmbedder):
    """
    Class for parallel processing images using the CLIP model.
    """

    def __call__(self, image_paths: list, num_workers: int = 4) -> None:
        """
        Parallel processes images using the CLIP model.

        Args:
            image_paths (list): A list of paths to the images.
            num_workers (int): Number of worker processes for parallel processing.

        Returns:
            None
        """
        output_queue = Queue()
        processes = []
        for img_path in image_paths:
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
            image_embeddings.append(output_queue.get().cpu())
            save_embeddings(image_embeddings)

    def process_image(self, img_path: str, output_queue: Queue) -> None:
        """
        Worker function to process images in parallel.

        Args:
            img_path (str): Path to the image.
            output_queue (Queue): Queue to store the processed image embeddings.

        Returns:
            None
        """
        image = Image.open(img_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)
        output_queue.put(image_embedding)


if __name__ == "__main__":
    image_paths = get_image_paths()
    embd = ImageEmbedder()
    embd(image_paths)
