import re
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit
import os
import pandas as pd
from model import load_and_prepare, train_regression_models, derive_mental_wellness_label
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_KEY")
socketio = SocketIO(app)

client = genai.Client()

file_path = "data.csv"
df, encoders = load_and_prepare(file_path)
if df is not None:
    model_mental, model_addicted = train_regression_models(df)
    feature_columns = list(df.drop(columns=["Mental_Health_Score", "Addicted_Score"]).columns)
else:
    model_mental, model_addicted = None, None
    feature_columns = []

if df is not None:
    model_mental, model_addicted = train_regression_models(df)
else:
    model_mental, model_addicted = None, None

def remove_repeated_text(response):
    response = re.sub(r'\s+', ' ', response).strip()
    seen = set()
    filtered_words = []
    for word in response.split():
        if word not in seen:
            seen.add(word)
            filtered_words.append(word)
    return " ".join(filtered_words)

question_order = [
    "Age", "Gender", "Academic_Level", "Country", "Avg_Daily_Usage_Hours",
    "Most_Used_Platform", "Affects_Academic_Performance",
    "Relationship_Status", "Conflicts_Over_Social_Media", "Sleep_Hours_Per_Night"
]

question_texts = {
    "Age": "Enter your age (e.g., 20): ",
    "Gender": "Enter your gender (male/female/other): ",
    "Academic_Level": "Enter your academic level (e.g., undergraduate): ",
    "Country": "Enter your country: ",
    "Avg_Daily_Usage_Hours": "How many hours per day do you use social media (e.g., 4.5)? ",
    "Most_Used_Platform": "Which social media platform do you use the most? ",
    "Affects_Academic_Performance": "Does social media affect your academic performance? (yes/no): ",
    "Relationship_Status": "What is your relationship status? (e.g., single): ",
    "Conflicts_Over_Social_Media": "How many conflicts have you had over social media recently (0-10)? ",
    "Sleep_Hours_Per_Night": "How many hours do you sleep per night? "
}

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("message")
def handle_message(msg):
    print(f"Received message: {msg}")

    if msg.strip() == "@start":
        session['responses'] = {}
        session['question_index'] = 0
        emit("response", "Welcome to SociaLytix! Let's assess your mental wellness based on your social media use. Please answer honestly.")
        emit("response", question_texts[question_order[0]])
        print("Sent first question")
        return

    if "question_index" in session:
        q_index = session["question_index"]
        if q_index < len(question_order):
            current_q = question_order[q_index]
            session['responses'][current_q] = msg.strip()
            session["question_index"] += 1
            print(f"Stored response for {current_q}: {msg.strip()}")

            if session["question_index"] < len(question_order):
                next_q = question_order[session["question_index"]]
                emit("response", question_texts[next_q])
                print(f"Sent next question: {next_q}")
            else:
                print("All questions answered, making predictions...")
                try:
                    input_data = session["responses"]
                    input_df = pd.DataFrame([input_data])
                    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

                    for col in encoders:
                        if col in input_df.columns:
                            val = input_df.at[0, col]
                            if val in encoders[col].classes_:
                                input_df.at[0, col] = encoders[col].transform([val])[0]
                            else:
                                input_df.at[0, col] = -1

                    for num_col in ["Age", "Avg_Daily_Usage_Hours", "Conflicts_Over_Social_Media", "Sleep_Hours_Per_Night"]:
                        if num_col in input_df.columns:
                            input_df[num_col] = pd.to_numeric(input_df[num_col], errors='coerce').fillna(0)

                    pred_mental = model_mental.predict(input_df)[0]
                    pred_addicted = model_addicted.predict(input_df)[0]
                    sleep_hours = float(input_df["Sleep_Hours_Per_Night"].values[0])
                    conflicts = int(input_df["Conflicts_Over_Social_Media"].values[0])
                    label = derive_mental_wellness_label(pred_mental, pred_addicted, sleep_hours, conflicts)

                    emit("response", f"Predicted Mental Health Score: {pred_mental:.1f}")
                    emit("response", f"Predicted Addiction Score: {pred_addicted:.1f}")
                    emit("response", f"Mental Wellness Label: {label}")
                    print("Prediction sent to client.")
                    prompt = (
                        "You are SociaLytix, a supportive and friendly AI designed to help users reflect on their social media habits "
                        "and their impact on mental well-being. Offer 2-3 friendly, actionable suggestions if needed. "
                        "Keep your tone warm, non-judgmental, and encouraging. Never request sensitive or identifying personal details. "
                        "End your analysis with a motivational message like, “Take care of yourself — you're doing better than you think. 🌱”. "
                        "Don't reply in Readme format. Keep your response short and conversational. "
                        "Now respond to the user message:\n" + f"Accuracy: {pred_mental:.1f}, Addiction: {pred_addicted:.1f}, Label: {label}. "
                    )
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        ),
                    )
                    response_text = response.candidates[0].content.parts[0].text
                    emit("response", response_text)
                except Exception as e:
                    print(f"Prediction error: {e}")
                    emit("response", f"Error during prediction: {str(e)}")

                session.pop("question_index", None)
                session.pop("responses", None)
        return

    try:
        prompt = (
            "You are SociaLytix, a supportive and friendly AI designed to help users reflect on their social media habits "
            "and their impact on mental well-being. Offer 2-3 friendly, actionable suggestions if needed. "
            "Keep your tone warm, non-judgmental, and encouraging. Never request sensitive or identifying personal details. "
            "End your analysis with a motivational message like, “Take care of yourself — you're doing better than you think. 🌱”. "
            "Don't reply in Readme format. Keep your response short and conversational. "
            "Now respond to the user message:\n" + msg.strip()
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        response_text = response.candidates[0].content.parts[0].text
        emit("response", response_text)
    except Exception as e:
        print(f"Gemini error: {e}")
        emit("response", "Oops, I had trouble thinking that through. Mind trying again?")
    
if __name__ == "__main__":
    socketio.run(app, debug=True)
