import cv2
from deepface import DeepFace
import pyttsx3
import time
from collections import deque

# Text-to-speech init
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Emotion buffer to reduce flickering
emotion_buffer = deque(maxlen=5)  # store last 5 emotions
last_stable_emotion = None
last_spoken_time = 0

# Webcam
cap = cv2.VideoCapture(0)

# Emotion to Emoji map (we’ll only print as text now)
emoji_map = {
    "happy": "Happy 😊",
    "sad": "Sad 😢",
    "angry": "Angry 😠",
    "surprise": "Surprised 😲",
    "fear": "Fear 😨",
    "disgust": "Disgust 😒",
    "neutral": "Neutral 😐"
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    try:
        # Resize for speed
        small_frame = cv2.resize(frame, (640, 480))

        result = DeepFace.analyze(small_frame, actions=['emotion'],
                                  enforce_detection=False, detector_backend='opencv')

        emotion = result[0]['dominant_emotion']
        emotion_buffer.append(emotion)

        # Get most frequent emotion in buffer
        stable_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

        label = emoji_map.get(stable_emotion, stable_emotion.capitalize())

        # Display text on screen
        cv2.putText(frame, f"Emotion: {label}", (50, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 0), 3)

        # Speak only if stable emotion changed & 3 sec passed
        if stable_emotion != last_stable_emotion and (time.time() - last_spoken_time) > 3:
            engine.say(f"You look {stable_emotion}")
            engine.runAndWait()
            last_stable_emotion = stable_emotion
            last_spoken_time = time.time()

    except Exception as e:
        cv2.putText(frame, "No face detected", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

    cv2.imshow("Stable Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
