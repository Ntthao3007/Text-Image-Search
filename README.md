# Text-Image-Search

The goal of this project is to develop a micro-system for image search that allows users to search for images based on prompts.

####  Dataset #### 

The Unplash Dataset [link](https://github.com/unsplash/datasets)

To download the dataset and preproces:
* ```python ./TextSearch/data/preprocess```

####  Image Embedding Extractor #### 

To extract the embeddings from the images and store them:
* ```python ./TextSearch/src/image_encoder.py```

#### Query Embedding Extractor #####

To extract the embeddings from the query:
* ```python ./TextSearch/src/query_encoder.py --query "a beach"```

#### Search Images from Query ####

To search similar images based on the query:
* ```python ./TextSearch/src/search.py --query "a beach" --top_k 5```

####  To use Gradio Demo #### 
* ```python ./TextSearch/src/app.py```

####  Tools #### 
* ```pip install -r requirements.txt ``` 
