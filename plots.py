import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

accuracies = {}

for name, model in models.items():
    y_pred = model.predict(X_test) 
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="viridis")
plt.ylim(0, 1) 
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()


sample_image = X_test[0].reshape((90, 90))

hog_features, hog_image = hog(sample_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sample_image, cmap="gray")
plt.title("Original Grayscale Image")

plt.subplot(1, 2, 2)
plt.imshow(hog_image, cmap="gray")
plt.title("HOG Feature Visualization")

plt.show()
