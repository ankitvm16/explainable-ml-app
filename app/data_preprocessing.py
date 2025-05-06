import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, target_column):
    df = df.copy()
    le_dict = {}
    for column in df.select_dtypes(include=['object']).columns:
        if column != target_column:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            le_dict[column] = le
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42), le_dict
