import os
from huggingface_hub import HfApi

# Cấu hình thông tin
TOKEN = "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"
REPO_ID = "ManhQuangAI/workspace-deepFake"
FOLDER_PATH = "/workspace"
REPO_TYPE = "dataset" # Chọn: "model", "dataset", hoặc "space"

print(f"Đang chuẩn bị upload thư mục {FOLDER_PATH} lên {REPO_ID}...")

api = HfApi(token=TOKEN)

# 1. Tự động tạo Repo Public nếu chưa tồn tại
api.create_repo(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    private=False, # False là Public
    exist_ok=True
)

# 2. Upload toàn bộ thư mục
api.upload_folder(
    folder_path=FOLDER_PATH,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)

print("Upload thành công!")
