from sklearn.metrics import classification_report

def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n🔹 {name} Model Evaluation")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
evaluate_models(models, X_test, y_test)
