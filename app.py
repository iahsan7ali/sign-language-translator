import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pickle

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class SignLanguageFinal(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageFinal, self).__init__()
        # hidden_size MUST be 128 to match the 512-sized tensors in your error
        self.lstm = nn.LSTM(225, 128, batch_first=True, num_layers=2, bidirectional=True)
        
        # The FC layer must take 128 * 2 (for bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Pull the last time step from the bidirectional output
        out = self.fc(lstm_out[:, -1, :])
        return out

# --- 2. PREPROCESSING LOGIC ---
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# --- 3. LOAD DATA & MODEL ---
@st.cache_resource
def load_resources():
    # Load Labels
    with open('label_map.pkl', 'rb') as f:
        actions = pickle.load(f)
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageFinal(num_classes=len(actions)).to(device)
    model.load_state_dict(torch.load('best_sign_model_200.pth', map_location=device))
    model.eval()
    return model, actions, device

model, actions, device = load_resources()

# --- 4. REAL-TIME ENGINE ---
class SignTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_holistic = mp_holistic.Holistic(
         min_detection_confidence=0.5,
         min_tracking_confidence=0.5
          )
        self.sequence = []
        self.prediction = "..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_holistic.process(image_rgb)
        
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:] # Last 30 frames
        
        if len(self.sequence) == 30:
            input_data = torch.tensor([self.sequence], dtype=torch.float32).to(device)
            with torch.no_grad():
                res = model(input_data)
                idx = torch.argmax(res).item()
                self.prediction = actions[idx]

        cv2.putText(img, f'SIGN: {self.prediction}', (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Sign Language AI", layout="wide")
st.title("🤟 Real-Time Sign Language Translator")
st.sidebar.info("This model recognizes 200 signs with 82.9% accuracy.")

webrtc_streamer(
    key="sign_lang", 
    video_processor_factory=SignTransformer, # Updated argument name
    rtc_configuration={ 
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
