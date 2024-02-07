from embedder import TextEmbedder
from qdrant_client_upload import UploadQdrant
from utils import get_image_paths, parse_args
from config import COLLECTION_NAME
from PIL import Image


def find_best_matches(query, embedder, qdclient, limit):
    query_embedding = embedder([query])
    search_result = qdclient.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.squeeze().tolist(),
        limit=limit,
    )
    hit_paths = list()
    for hit in search_result:
        hit_paths = [hit.payload["path"] for hit in search_result]
        # hit_paths.append(hit_path)
    return hit_paths


def show_results(hits):
    for hit in enumerate(hits):
        print(hit)
        image = Image.open(hit)
        image.show()



def main():
    args = parse_args()
    image_paths = get_image_paths()
    query_embedder = TextEmbedder()

    # create qdrant client
    qdclient = UploadQdrant()

    # only if collection is not uploaded
    if COLLECTION_NAME not in [
        col.name for col in qdclient.client.get_collections().collections
    ]:
        # upload the image embeddings
        qdclient.upload(image_paths, args)

    # take query
    query = "apple"
    # produce results
    search_hits = find_best_matches(
        query=query,
        embedder=query_embedder,
        qdclient=qdclient.client,
        limit=args.hit_limit,
    )
    show_results(search_hits)


if __name__ == "__main__":
    main()
