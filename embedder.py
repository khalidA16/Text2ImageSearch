from typing import Any
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils import get_image_paths, save_embeddings
from tqdm import tqdm


class CustomImageDataset(Dataset):
    """
    Creates a custom image dataset.
    """

    def __init__(self, image_paths, processor) -> None:
        """
        Initializes the CustomImageDataset.

        Args:
            image_paths (list): A list of paths to the images.
            processor ([`CLIPImageProcessor`], *optional*): processor to apply transforms

        Returns:
            None
        """
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self) -> int:
        """
        Returns:
            number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Retrieves an image and processes it.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tensor: Processed image tensor.
        """
        img = self.image_paths[idx]
        image = self.processor(images=Image.open(img), return_tensors="pt")
        return image["pixel_values"].squeeze()


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

    def __call__(self, image_paths: list, batch_size: int = 32) -> None:
        """
        Embeds images using the CLIP model.

        Args:
            image_paths (list): A list of paths to the images.

        Returns:
            None
        """
        self.batch_size = batch_size

        my_dataset = CustomImageDataset(image_paths, self.processor)
        dataloader = DataLoader(my_dataset, batch_size=self.batch_size, shuffle=False)

        num_batches = len(dataloader)
        image_embeddings = list()
        with tqdm(total=num_batches) as pbar:
            for batch in dataloader:
                with torch.no_grad():
                    image_features = self.model.get_image_features(
                        pixel_values=batch.to(self.device)
                    )
                image_embeddings.append(image_features)
                pbar.update(1)
        image_embeddings = torch.cat(image_embeddings, dim=0).cpu()
        save_embeddings(image_embeddings.tolist())


if __name__ == "__main__":
    image_paths = get_image_paths()
    embedder = ImageEmbedder()
    embedder(image_paths)
