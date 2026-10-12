import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
from pose_engine import process_exercise

st.set_page_config(page_title="Sport Position Scanner", page_icon="⚡", layout="wide")

# تنسيق الواجهة الرياضية
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .sports-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #374151;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚡ Sport Scanner")
    lang = st.selectbox("Language / اللغة", ["العربية", "English"])
    exercise = st.selectbox("Exercise / التمرين", ["Squat", "Push-up", "Plank"])
    st.info("اضغط على Start في نافذة الكاميرا للبدء." if lang == "العربية" else "Click Start in the video window to begin.")

is_ar = (lang == "العربية")
st.title("⚡ Sport Position Scanner")
st.caption("نظام التتبع الرياضي الفوري" if is_ar else "Real-Time Fitness Tracking")

# إعداد خوادم الاتصال لتعمل الكاميرا بسلاسة على الجوال
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# الاستخدام الحديث المعتمد للنسخ الجديدة من ميديا بايب
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.IMAGE
)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        with PoseLandmarker.create_from_options(options) as landmarker:
            detection_result = landmarker.detect(mp_image)

            if detection_result.pose_landmarks:
                for landmarks in detection_result.pose_landmarks:
                    is_correct, angle, feedback = process_exercise(exercise, landmarks)
                    color = (0, 255, 0) if is_correct else (0, 0, 255)
                    cv2.putText(rgb, f"Angle: {angle}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(rgb, feedback, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return rgb

# تشغيل مشغل الكاميرا السحابي المناسب للجوال والكمبيوتر
webrtc_streamer(
    key="sport-scanner",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False}
)
