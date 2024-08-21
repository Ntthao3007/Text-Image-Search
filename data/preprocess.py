import os
from argparse import ArgumentParser
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

DOWNLOAD_URL = "https://unsplash-datasets.s3.amazonaws.com/lite/latest/unsplash-research-dataset-lite-latest.zip"
DATASET_PATH = Path(__file__).resolve().parents[1] / "data"
DOWNLOADED_PHOTOS_PATH = DATASET_PATH / "photos"
FEATURE_PATH = DATASET_PATH / "features"
DOWNLOADED_PHOTOS_PATH.mkdir(parents=True, exist_ok=True)
FEATURE_PATH.mkdir(parents=True, exist_ok=True)

def setup_parser():
    parser = ArgumentParser()
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--threads_count", type=int, default=128)
    parser.add_argument("--num_images", type=int, default=3000)
    return parser

def download_photo(image_width, photo):
    photo_id = photo[0]
    photo_url = photo[1] + f"?w={image_width}"
    photo_path = DOWNLOADED_PHOTOS_PATH / f"{photo_id}.jpg"
    if not photo_path.exists():
        try:
            urlretrieve(photo_url, photo_path)
        except Exception as e:
            print(f"Cannot download {photo_url}: {e}")
            pass

def download_images(photos, image_width, threads_count=128):
    print("Photo downloading begins...")
    pool = ThreadPool(threads_count)
    pool.map(partial(download_photo, image_width), photos)
    print(f'Photos downloaded: {len(photos)}')
    print("Photo downloading finished!")

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    zip_filename = "unsplash-dataset.zip"
    print(f"Downloading metadata file {zip_filename}...")
    os.system(f"curl -o {zip_filename} {DOWNLOAD_URL}")
    print(f"Extracting {zip_filename}...")
    os.system(f"unzip {zip_filename} -d {str(DATASET_PATH)}")

    photos = pd.read_csv(DATASET_PATH / "photos.tsv000", sep='\t', header=0)
    photo_urls = photos[['photo_id', 'photo_image_url']].values.tolist()[:args.num_images]

    download_images(photo_urls, args.image_width, args.threads_count)
    photos_files = list(DOWNLOADED_PHOTOS_PATH.glob("*.jpg"))
    