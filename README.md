# Stable-Emotion-Detector
🔊 Real-time emotion detection using DeepFace and OpenCV with voice feedback via pyttsx3. Minimizes flickering using an emotion buffer for stable results.
🗂️ Folder Structure

Stable-Emotion-Detector/
│
├── emotion.py
├── requirements.txt
├── README.md
└── .gitignore
📄 main.py
This is the Python code you provided. Save it as main.py.

📦 requirements.txt
txt
Copy
Edit
opencv-python
deepface
pyttsx3
📘 README.md
markdown
Copy
Edit
# 🧠 Stable Emotion Detector

A real-time emotion recognition system using DeepFace, OpenCV, and pyttsx3.  
It detects the dominant emotion from your face via webcam and speaks it out loud using text-to-speech.

---

## 💡 Features

- Real-time emotion detection
- Emoji and label display on screen
- Voice feedback when emotion changes
- Flicker reduction using emotion buffer

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
🚀 How to Run
bash
Copy
Edit
python main.py
Press q to quit the app.

🧰 Tech Stack
DeepFace

OpenCV

pyttsx3
