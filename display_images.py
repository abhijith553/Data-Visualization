folders = ['Pre', 'Benign', 'Early', 'Pro']
path = '/root/.cache/kagglehub/datasets/mehradaria/leukemia/versions/1/Original/'
print("Avaialable folders are:")
for j in folders:
  print(j)
choice = input("Enter required folder name: ")
path2 = path + choice
files = os.listdir(path2)
print(files[:5]) # lists the names of first 5 files

import cv2
import matplotlib.pyplot as plt

file_no = 0   # first image at 0, last at n

image_path = os.path.join(path2, files[file_no]) # change value to access different jpg's
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display
plt.imshow(image)
plt.title(f"Current image: {files[file_no]}")
plt.show()

# display segmented images

folders = ['Pre', 'Benign', 'Early', 'Pro']
path = '/root/.cache/kagglehub/datasets/mehradaria/leukemia/versions/1/Segmented/'
print("Avaialable folders are:")
for j in folders:
  print(j)
choice = input("Enter required folder name: ")
path2 = path + choice
files = os.listdir(path2)
print(files[:5])

import cv2
import matplotlib.pyplot as plt

file_no = 0

image_path = os.path.join(path2, files[file_no]) 
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.title(f"Current image: {files[file_no]}")
plt.show()
