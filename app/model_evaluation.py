from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return report, matrix, accuracy
