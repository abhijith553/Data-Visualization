'''

    Convert to HSV color space
    Apply color thresholding
    Remove noise morphological operations
    Find boundaries of the detected cells (countour detection)
    Feature extraction (for machine learning)
    Save pre processed image

'''

import os


base_path = "/content/drive/MyDrive/leukemia-dataset/"


pre_path = os.path.join(base_path, "Pre")
benign_path = os.path.join(base_path, "Benign")
early_path = os.path.join(base_path, "Early")
pro_path = os.path.join(base_path, "Pro")


pre_files = os.listdir(pre_path) if os.path.exists(pre_path) else []
benign_files = os.listdir(benign_path) if os.path.exists(benign_path) else []
early_files = os.listdir(early_path) if os.path.exists(early_path) else []
pro_files = os.listdir(pro_path) if os.path.exists(pro_path) else []


print(f"Pre: {len(pre_files)} files")
print(f"Benign: {len(benign_files)} files")
print(f"Early: {len(early_files)} files")
print(f"Pro: {len(pro_files)} files")


print("\nPre files:", pre_files[:5])
print("Benign files:", benign_files[:5])
print("Early files:", early_files[:5])
print("Pro files:", pro_files[:5])

import cv2
import matplotlib.pyplot as plt

path_to_image = '/content/drive/MyDrive/leukemia-dataset/Benign/WBC-Benign-001.jpg'


image = cv2.imread(path_to_image)

if image is None:
    print("Error: Unable to load image. Check the file path.")
else:

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
    plt.title("HSV Image")
    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt


path_to_image = '/content/drive/MyDrive/leukemia-dataset/Benign/WBC-Benign-001.jpg'


image = cv2.imread(path_to_image)


hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower_bound = np.array([120, 50, 50])  # Lower HSV threshold for purple/blue
upper_bound = np.array([160, 255, 255])  # Upper HSV threshold for purple/blue


mask = cv2.inRange(hsv_image, lower_bound, upper_bound)


segmented = cv2.bitwise_and(image, image, mask=mask)


plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title("Color Segmented Image")
plt.axis("off")
plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

root_dir = "/content/drive/MyDrive/leukemia-dataset/"


folders = ["Pre", "Benign", "Early", "Pro"]


output_dir = "/content/drive/MyDrive/leukemia-dataset/New_Segmented_Images/"
os.makedirs(output_dir, exist_ok=True)


lower_bound = np.array([120, 50, 50]) 
upper_bound = np.array([160, 255, 255])  


for folder in folders:
    folder_path = os.path.join(root_dir, folder)
    segmented_folder_path = os.path.join(output_dir, folder)

    os.makedirs(segmented_folder_path, exist_ok=True)


    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)

        image = cv2.imread(file_path)
        if image is None:
            print(f"Error loading image: {file_path}")
            continue

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)


        segmented = cv2.bitwise_and(image, image, mask=mask)


        segmented_file_path = os.path.join(segmented_folder_path, file)
        cv2.imwrite(segmented_file_path, segmented)


        print(f"Segmented and saved: {segmented_file_path}")

print("Segmentation complete for all images.")

