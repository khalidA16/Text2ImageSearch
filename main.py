from embedder import ImageEmbedder
from qdrant_client_upload import UploadQdrant
from utils import get_image_paths, parse_args
import subprocess


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
