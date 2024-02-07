#!/bin/bash

YELLOW='\033[1;33m'
NC='\033[0m'

# ------ Python Virtual environment ------

# create virtual environment
python3 -m venv venv

# activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${YELLOW}Virtual environment activated.${NC}"

# install libraries
pip install -r requirements.txt

# ------ Prepare dataset ------

mkdir image_dataset

# download datasets
echo -e "${YELLOW}Downloading image dataset${NC}"
wget https://storage.googleapis.com/ads-dataset/subfolder-0.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-1.zip

# unzip datasets
echo -e "${YELLOW}Unzipping datasets and moving it to 'image_dataset${NC}'"
unzip subfolder-0.zip -d image_dataset
unzip subfolder-1.zip -d image_dataset

# empty subdirectories into one directory
find ./image_dataset/ -type f -print0 | xargs -0 mv -t ./image_dataset

# remove empty subdirectories
rmdir ./image_dataset/0 && rmdir ./image_dataset/1
