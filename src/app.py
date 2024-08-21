import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from query_encoder import QueryEncoder
from search import PhotoSearcher

def display_results(query, count):
    query_encoder = QueryEncoder()
    searcher = PhotoSearcher(top_k=count)
    encoded_query_features = query_encoder.encode_search_query(query=query)
    results = searcher.search_photos(encoded_query_features)
    image_urls = [f"{url}?w=320" for url in results] 
    return image_urls

interface = gr.Interface(
    fn=display_results,
    inputs=[
        gr.Textbox(label="Search Query"),
        gr.Slider(minimum=1, maximum=10, value=3, label="Number of Results")
    ],
    outputs=gr.Gallery(label="Search Results"),
    title="Image Search"
)

interface.launch()