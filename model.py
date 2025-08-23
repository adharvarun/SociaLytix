import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import root_mean_squared_error

def derive_mental_wellness_label(mental_score, addicted_score, sleep_hours, conflicts):
    score = 0
    if mental_score >= 8:
        score += 2
    elif mental_score >= 6:
        score += 1

    if addicted_score <= 3:
        score += 2
    elif addicted_score <= 6:
        score += 1

    if sleep_hours >= 7:
        score += 2
    elif sleep_hours >= 6:
        score += 1

    if conflicts == 0:
        score += 2
    elif conflicts <= 2:
        score += 1

    if score >= 7:
        return "Healthy"
    elif score >= 4:
        return "Moderate"
    else:
        return "Unwell"

def load_and_prepare(file_path):
    df = pd.read_csv(file_path)

    df.drop(columns=[col for col in ['Student_ID', 'id'] if col in df.columns], inplace=True)

    categorical_columns = ["Age", "Gender", "Academic_Level", "Country", "Avg_Daily_Usage_Hours",
                           "Most_Used_Platform", "Affects_Academic_Performance",
                           "Relationship_Status", "Conflicts_Over_Social_Media", "Sleep_Hours_Per_Night"]

    for col in categorical_columns + ["Mental_Health_Score", "Addicted_Score"]:
        df[col] = df[col].astype(str).str.lower()

    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    imputer = KNNImputer(n_neighbors=3)
    df.iloc[:, :] = imputer.fit_transform(df)

    return df, encoders

def train_regression_models(df):
    X = df.drop(columns=["Mental_Health_Score", "Addicted_Score"])
    y_mental = df["Mental_Health_Score"].astype(float)
    y_addicted = df["Addicted_Score"].astype(float)

    X_train, X_test, y_mental_train, y_mental_test = train_test_split(X, y_mental, test_size=0.2, random_state=42)
    _, _, y_addicted_train, y_addicted_test = train_test_split(X, y_addicted, test_size=0.2, random_state=42)

    model_mental = RandomForestRegressor(n_estimators=100, random_state=42)
    model_addicted = RandomForestRegressor(n_estimators=100, random_state=42)

    model_mental.fit(X_train, y_mental_train)
    model_addicted.fit(X_train, y_addicted_train)

    pred_mental = model_mental.predict(X_test)
    pred_addicted = model_addicted.predict(X_test)

    print("\n--- Regression Performance ---")
    print(f"Mental Health Score RMSE: {root_mean_squared_error(y_mental_test, pred_mental):.2f}")
    print(f"Addicted Score RMSE: {root_mean_squared_error(y_addicted_test, pred_addicted):.2f}")

    return model_mental, model_addicted

def ask_user_inputs(encoders):
    user_data = {}
    questions = {
        "Age": "Enter your age (e.g., 20): ",
        "Gender": "Enter your gender (e.g., male/female): ",
        "Academic_Level": "Enter your academic level (e.g., undergraduate): ",
        "Country": "Enter your country: ",
        "Avg_Daily_Usage_Hours": "How many hours do you use social media per day (e.g., 4.5)? ",
        "Most_Used_Platform": "Which social media platform do you use the most? ",
        "Affects_Academic_Performance": "Does social media affect your academic performance? (yes/no): ",
        "Sleep_Hours_Per_Night": "How many hours do you sleep per night? ",
        "Relationship_Status": "What is your relationship status? (e.g., single): ",
        "Conflicts_Over_Social_Media": "How many conflicts have you had over social media recently (0-10)? "
    }

    for col, prompt in questions.items():
        value = input(prompt).strip().lower()
        if col in encoders and value in encoders[col].classes_:
            user_data[col] = encoders[col].transform([value])[0]
        elif col in encoders:
            user_data[col] = -1
        else:
            try:
                user_data[col] = float(value)
            except ValueError:
                user_data[col] = 0.0
    return pd.DataFrame([user_data])

def main():
    df, encoders = load_and_prepare("data.csv")
    model_mental, model_addicted = train_regression_models(df)

    user_input_df = ask_user_inputs(encoders)

    pred_mental = model_mental.predict(user_input_df)[0]
    pred_addicted = model_addicted.predict(user_input_df)[0]
    sleep_hours = float(user_input_df["Sleep_Hours_Per_Night"].values[0])
    conflicts = int(user_input_df["Conflicts_Over_Social_Media"].values[0])

    label = derive_mental_wellness_label(pred_mental, pred_addicted, sleep_hours, conflicts)

    print("\n--- Prediction Results ---")
    print(f"Predicted Mental Health Score: {pred_mental:.1f}")
    print(f"Predicted Addicted Score: {pred_addicted:.1f}")
    print(f"Predicted Mental Wellness Label: {label}")