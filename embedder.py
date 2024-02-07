from typing import Any
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.multiprocessing as mp
from queue import Queue
from utils import get_image_paths, save_embeddings, parse_args


class Embedder:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


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


class ImageEmbedder(Embedder):

    def __init__(self):
        super().__init__()
        self.mode = "image"

    def __call__(self, image_paths,batch_size):
        embeddings = list()
        for img_path in image_paths:
            inputs = self.processor(
                images=Image.open(img_path), return_tensors="pt"
            ).to(self.device)
        with torch.no_grad():
            embeddings.append(self.model.get_image_features(**inputs).cpu())
        save_embeddings(embeddings, self.mode)
        return embeddings

        # batch processing of images
        # embeddings = BatchProcessImages(images_paths,batch_size)
        # return embeddings

        # Parallel processing of images
        # embeddings = ParallelProcessImages(images_paths)
        # return embeddings


class BatchProcessImages(ImageEmbedder):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def __call__(self, image_paths):
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

            save_embeddings(image_embeddings, self.mode)
            return image_embeddings


class ParallelProcessImages(ImageEmbedder):
    # Process images in parallel using multiprocessing

    def __call__(self, image_paths, num_workers=4):
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
            save_embeddings(image_embeddings, self.mode)
        return image_embeddings

    def process_image(self, img_path, output_queue):
        # Define a worker function to process images
        image = Image.open(img_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)
        output_queue.put(image_embedding)


if __name__ == "__main__":
    args = parse_args()
    image_paths = get_image_paths()
    embd = ImageEmbedder()
    embd(image_paths,args.batch_size)
