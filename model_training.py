import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def load_images_and_labels(base_path):
    labels = []
    features = []
    categories = ["Pre", "Benign", "Early", "Pro"]

    for category in categories:
        folder_path = os.path.join(base_path, category)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            image = cv2.imread(img_path)
            if image is None:
                continue


            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            resized = cv2.resize(gray, (128, 128))

            hog_features, _ = hog(resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

            features.append(hog_features)
            labels.append(category)

    return np.array(features), np.array(labels)


dataset_path = "/content/drive/MyDrive/leukemia-dataset/New_Segmented_Images"

X, y = load_images_and_labels(dataset_path)


le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
