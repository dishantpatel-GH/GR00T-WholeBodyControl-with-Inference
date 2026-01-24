import os
from huggingface_hub import HfApi, upload_folder
from tqdm import tqdm

# -----------------------------
# CHANGE THESE
# -----------------------------
HF_USERNAME = "dishantpatel1207"
REPO_NAME = "pick_place_2"
LOCAL_FOLDER = "outputs/2026-01-13-10-16-40-G1-sim/"
PRIVATE = False   # set True if you want private dataset
# -----------------------------

repo_id = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi()

# 1. Create dataset repo (if it doesn't exist)
print("Creating (or reusing) Hugging Face dataset repo...")
api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    private=PRIVATE,
    exist_ok=True
)

# 2. Upload entire folder
print("Uploading dataset... This may take a while ⏳")

upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="",  # upload at root
)

print("\n✅ Upload complete!")
print(f"🌍 Your dataset is now live at: https://huggingface.co/datasets/{repo_id}")
