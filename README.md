<img src="./static/Logo.png" alt="Logo" width="100px" height="100px" />

# SociaLytix 🧠📱

**AI-powered analyser tool that predicts addiction and mental health scores based on social media usage and lifestyle factors.**  

SociaLytix combines **machine learning** (Random Forest Regressors) with an interactive **Flask + Socket.IO web app** to provide users with insights into their **mental wellness**. The system predicts a **Mental Health Score**, **Addiction Score**, and a categorical **Wellness Label** (`Healthy`, `Moderate`, or `Unwell`). It also integrates **Google Gemini AI** to deliver personalized, supportive feedback.  

---

## ✨ Features  
- Predicts **Mental Health Score** and **Addiction Score** using trained regression models.  
- Assigns a **Wellness Label** based on user habits (sleep, conflicts, addiction, stress levels).  
- Interactive **chat-style questionnaire** powered by Flask-SocketIO.  
- AI-driven **personalized feedback** using Google Gemini (`gemini-2.5-flash`).  
- Handles missing or noisy data via **KNN Imputation**.  
- User-friendly **web interface** with conversational flow.  

---

## 🏗️ Project Structure  

```
.
├── app.py              # Flask web app + chat logic
├── model.py            # ML training, preprocessing, prediction pipeline
├── templates/
│   └── index.html      # Web UI (chat interface)
├── static/
│   └── Logo.png        # Project logo
├── data.csv            # Dataset (user factors & scores)
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

---

## ⚙️ Installation  

1. **Clone the repo**  
```bash
git clone https://gitlab.com/adharvarun-projects/SociaLytix.git
cd SociaLytix
```

2. **Create a virtual environment & install dependencies**  
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt
```

3. **Set environment variables**  
Create a `.env` file in the root directory:  
```env
FLASK_KEY=your_secret_key
GEMINI_API_KEY=your_gemini_api_key
```

4. **Run the app**  
```bash
python app.py
```

5. Open your browser at **http://localhost:5000** 🚀  

---

## 🧠 How It Works  

1. **Data Preparation**  
   - Cleans and encodes categorical features (Age, Gender, Country, etc.).  
   - Missing values filled using **KNN Imputer**.  

2. **Model Training**  
   - Random Forest regressors trained separately for **Mental Health** and **Addiction** scores.  
   - Performance evaluated using **RMSE**.  

3. **User Interaction**  
   - Users answer a series of lifestyle & social media usage questions.  
   - The trained models generate predictions.  
   - A **wellness label** is assigned using `derive_mental_wellness_label()`.  

4. **AI Feedback**  
   - Google Gemini provides warm, non-judgmental suggestions.  
   - Example: *"Try setting screen-free hours before bedtime. 🌙 You're doing better than you think!"*  

---

## 📊 Example Prediction  

```
Predicted Mental Health Score: 6.8
Predicted Addiction Score: 4.2
Mental Wellness Label: Moderate
AI Feedback: "You're balancing things okay, but reducing late-night scrolling might help. 🌱"
```

---

## 📦 Requirements  

```
flask
flask-socketio
pandas
numpy
scikit-learn
python-dotenv
google-genai
```

---

## 🚧 Future Improvements  
- Add a **dashboard** with visual analytics.
- Expand dataset for better generalization.  
- Deploy to **Heroku / Render / Vercel** for public use.

---

## ⚠️ Disclaimer  
This tool is for **educational and awareness purposes only**. It is **not a medical diagnostic tool**.  
For professional advice, please consult a licensed mental health specialist.  
