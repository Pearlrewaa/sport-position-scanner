import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
from pose_engine import process_exercise

st.set_page_config(page_title="Sport Position Scanner", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
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

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        with PoseLandmarker.create_from_options(options) as landmarker:
            detection_result = landmarker.detect(mp_image)

            if detection_result.pose_landmarks:
                for landmarks in detection_result.pose_landmarks:
                    # استدعاء دالة التحقق من ملف pose_engine
                    try:
                        is_correct, angle, feedback = process_exercise(exercise, landmarks)
                    except Exception as e:
                        is_correct, angle, feedback = False, 0, "Error in engine"

                    # منطق العدّات
                    if exercise in ["Squat", "Push-up"]:
                        if angle > 150:
                            self.stage = "up"
                        if angle < 90 and self.stage == "up":
                            self.stage = "down"
                            self.counter += 1

                    # تحديد اللون (أخضر للحركة الصحيحة، أحمر للخاطئة)
                    color = (0, 255, 0) if is_correct else (0, 0, 255)
                    
                    # رسم مستطيل خلف النص لتوضيح الحالة (إضاءة تنبيهية على الشاشة)
                    overlay = rgb.copy()
                    cv2.rectangle(overlay, (20, 20), (600, 150), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, rgb, 0.4, 0, rgb)

                    # رسم النصوص والتقييم بوضوح تام
                    status_text = "CORRECT / صحيح" if is_correct else "INCORRECT / خطأ"
                    cv2.putText(rgb, f"Status: {status_text}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(rgb, f"Count: {self.counter} | Angle: {int(angle)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(rgb, str(feedback), (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return rgb

webrtc_streamer(
    key="sport-scanner",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False}
)
