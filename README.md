# Text2ImageSearch
### Installation
* Clone the repository
    ```
    git clone https://github.com/khalidA16/Text2ImageSearch.git
    ```
* Create virtual environment and download image dataset. The dataset can be found at `image_dataset`
    ```
    source setup.sh
    ```
### Run Qdrant
* Pull qdrant docker image 
    ```
    docker pull qdrant/qdrant
    ```
* Run qdrant 
    ```
    docker run -d -p 6333:6333 qdrant/qdrant
     ```
     `-d` to run container in detached mode
* Web UI access: http://0.0.0.0:6333/dashboard

