import streamlit as st
from config import COLLECTION_NAME
from embedder import TextEmbedder
from qdrant_client_upload import UploadQdrant
from typing import List

query_embedder = TextEmbedder()
qdclient = UploadQdrant().client


def find_best_matches(query: str, embedder, qdclient, limit: int) -> List[str]:
    """
    Finds the best matching images for a given query.

    Args:
        query (str): The query text.
        embedder: The embedding function to generate query embeddings.
        qdclient: The Qdrant client for searching image embeddings.
        limit (int): The maximum number of images to return.

    Returns:
        List[str]: A list of paths to the best matching images.
    """
    query_embedding = embedder([query])
    search_result = qdclient.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.squeeze().tolist(),
        limit=limit,
    )
    hit_paths = [hit.payload["path"] for hit in search_result]
    return hit_paths


def display_results(images: List[str]) -> None:
    """
    Displays images horizontally using columns layout.

    Args:
        images (List[str]): A list of paths to the images to be displayed.

    Returns:
        None
    """
    # Display images horizontally using columns layout
    st.subheader("Search Results")
    # Divide the screen into 3 columns
    col1, col2, col3 = st.columns(3)
    for idx, image in enumerate(images):
        if idx % 3 == 0:
            column = col1
        elif idx % 3 == 1:
            column = col2
        else:
            column = col3
        with column:
            st.image(image, width=150, caption=f"Image {idx+1}")


def main() -> None:
    """
    Main function to run the  Image Search Streamlite App.

    Returns:
        None
    """
    query = None
    st.markdown(
        "<h1 style='text-align: center;'>Text2Image Search App</h1>",
        unsafe_allow_html=True,
    )
    # Input query
    query = st.text_input("Enter your query:", key="query_input")
    num_images = st.number_input(
        "Number of images to show", min_value=1, max_value=10, value=5
    )

    if query:
        # find results from qdrant
        images = find_best_matches(query, query_embedder, qdclient, num_images)
        # display search results
        display_results(images)


if __name__ == "__main__":
    main()
