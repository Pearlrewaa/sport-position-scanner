import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
from pose_engine import process_exercise

st.set_page_config(
    page_title="Sport Position Scanner", 
    page_icon="⚡", 
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
        color: #f3f4f6;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚡ Sport Scanner Pro")
    st.markdown("---")
    lang = st.selectbox("🌐 Language / اللغة", ["العربية", "English"])
    exercise = st.selectbox("🎯 Select Exercise / اختر التمرين", ["Squat", "Push-up", "Plank"])
    st.markdown("---")
    st.info("اضغط على زر التشغيل (▶) داخل نافذة الكاميرا للبدء.")

is_ar = (lang == "العربية")

st.markdown('<p class="main-title">⚡ Sport Position Scanner</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">نظام التتبع الرياضي الفوري الذكي</p>' if is_ar 
    else '<p class="subtitle">Real-Time Fitness Pose Analysis System</p>', 
    unsafe_allow_html=True
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.IMAGE
)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.frame_count = 0  # لتخفيف الضغط ومنع التعليق

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.frame_count += 1
        # معالجة إطار واحد من كل ثلاثة إطارات لمنع التعليق تماماً
        if self.frame_count % 3 == 0:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            with PoseLandmarker.create_from_options(options) as landmarker:
                detection_result = landmarker.detect(mp_image)

                if detection_result.pose_landmarks:
                    for landmarks in detection_result.pose_landmarks:
                        try:
                            is_correct, angle, feedback = process_exercise(exercise, landmarks)
                        except Exception:
                            is_correct, angle, feedback = False, 0, "Ready..."

                        if exercise in ["Squat", "Push-up"]:
                            if angle > 150:
                                self.stage = "up"
                            if angle < 90 and self.stage == "up":
                                self.stage = "down"
                                self.counter += 1

                        color = (0, 255, 0) if is_correct else (0, 0, 255)
                        
                        # رسم لوحة التنبيهات الفورية
                        overlay = rgb.copy()
                        cv2.rectangle(overlay, (15, 15), <github_placeholder> (440, 125), (15, 23, 42), -1)
                        cv2.addWeighted(overlay, 0.7, rgb, 0.3, 0, rgb)
                        
                        border_color = (0, 255, 0) if is_correct else (0, 0, 255)
                        cv2.rectangle(rgb, (0, 0), (rgb.shape[1], rgb.shape[0]), border_color, 4)

                        status_str = "CORRECT / تمرير صحيح" if is_correct else "INCORRECT / عدل وضعيتك"
                        cv2.putText(rgb, f"Status: {status_str}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.putText(rgb, f"Count: {self.counter}  |  Angle: {int(angle)}", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(rgb, f"Tip: {feedback}", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return rgb

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    webrtc_streamer(
        key="sport-scanner",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False}
    )
