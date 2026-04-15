
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()
ROOT_DIR = Path("document-haystack")

# MAKE A .env file in the root of the REPO with the HUGGINGFACE_HUB_TOKEN
def download_data():
    """Download the document-haystack dataset if it is not already present.

    The dataset is stored under ``document-haystack`` and authenticated using
    the ``HUGGINGFACE_HUB_TOKEN`` environment variable.
    """
    if ROOT_DIR.exists() and any(ROOT_DIR.iterdir()):
        print(f"Skipping download, folder already exists: {ROOT_DIR}")
        return

    snapshot_download(
        "AmazonScience/document-haystack",
        repo_type="dataset",
        token=os.environ["HUGGINGFACE_HUB_TOKEN"],
        local_dir=str(ROOT_DIR),
    )

def clean_data():
    """Remove unwanted folders and files from the downloaded dataset.

    Keeps only inner-most folders named ``Text_TextNeedles`` and removes files
    whose names contain ``ImageNeedles``.
    """
    keep_name = "Text_TextNeedles"

    if not ROOT_DIR.exists():
        print(f"Nothing to clean, folder does not exist: {ROOT_DIR}")
        return

    # Delete all inner-most folders not called "Text_TextNeedles""
    for current_root, subdirs, _ in os.walk(ROOT_DIR, topdown=False):
        current_path = Path(current_root)
        if not subdirs and current_path.name != keep_name:
            shutil.rmtree(current_path)
            print(f"Deleted: {current_path}")

    # Delete all files with "ImageNeedles" in name
    for current_root, _, files in os.walk(ROOT_DIR):
        for file_name in files:
            if "ImageNeedles" in file_name:
                file_path = Path(current_root) / file_name
                file_path.unlink()
                print(f"Deleted file: {file_path}")

if __name__ == "__main__":
    download_data()
    clean_data()