import streamlit as st
from config import COLLECTION_NAME
from embedder import TextEmbedder
from qdrant_client_upload import UploadQdrant

query_embedder = TextEmbedder()
qdclient = UploadQdrant().client


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


def display_images(images):
    # Display images horizontally using columns layout
    st.subheader("Search Results")
    col1, col2, col3 = st.columns(3)  # Divide the screen into 3 columns
    for idx, image in enumerate(images):
        if idx % 3 == 0:
            column = col1
        elif idx % 3 == 1:
            column = col2
        else:
            column = col3
        with column:
            st.image(image, width=150, caption=f"Image {idx+1}")


def main():
    st.title("Image Search App")

    # Input query
    query = st.text_input("Enter your query:")
    num_images = st.number_input(
        "Number of images to show", min_value=1, max_value=10, value=5
    )

    # Button to trigger image search
    if st.button("Search"):
        images = find_best_matches(query, query_embedder, qdclient, num_images)
        if images:
            # Display images
            display_images(images)
        else:
            st.write("No images found matching the query.")


if __name__ == "__main__":
    main()
