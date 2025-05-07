import kagglehub

# Download latest version
path = kagglehub.dataset_download("mehradaria/leukemia")

print("Path to dataset files:", path)

from google.colab import drive
drive.mount('/content/drive')
# import shutil

# # Define Google Drive path
# drive_path = "/content/drive/MyDrive/leukemia-dataset-segmented"

# # Copy the dataset folder to Drive
# shutil.copytree('/root/.cache/kagglehub/datasets/mehradaria/leukemia/versions/1/Segmented', drive_path)

# print(f"Dataset saved to: {drive_path}")

import os
os.listdir('/root/.cache/kagglehub/datasets/mehradaria/leukemia/versions/1/Original')

PreCount = len(os.listdir('/root/.cache/kagglehub/datasets/mehradaria/leukemia/versions/1/Original/Pre'))
BenignCount = len(os.listdir('/root/.cache/kagglehub/datasets/mehradaria/leukemia/versions/1/Original/Benign'))
EarlyCount = len(os.listdir('/root/.cache/kagglehub/datasets/mehradaria/leukemia/versions/1/Original/Early'))
ProCount = len(os.listdir('/root/.cache/kagglehub/datasets/mehradaria/leukemia/versions/1/Original/Pro'))

print(PreCount, BenignCount, EarlyCount, ProCount)

