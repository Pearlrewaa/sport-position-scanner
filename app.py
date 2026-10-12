import streamlit as st
import cv2
import tempfile
import mediapipe as mp
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
    st.info("قومي برفع فيديو قصير لأدائك الرياضي، وسيقوم النظام بتحليله فوراً وإعطائك النتيجة!")

is_ar = (lang == "العربية")

st.markdown('<p class="main-title">⚡ Sport Position Scanner</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">نظام التتبع الرياضي الفوري وتحليل الأداء الذكي</p>' if is_ar 
    else '<p class="subtitle">Smart Fitness Pose Analysis System</p>', 
    unsafe_allow_html=True
)

# أداة رفع الفيديو
uploaded_file = st.file_uploader(
    "ارفعي فيديو التمرين هنا (MP4, MOV)..." if is_ar else "Upload exercise video here...", 
    type=["mp4", "mov", "avi"]
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.IMAGE
)

if uploaded_file is not None:
    # حفظ الفيديو المرفوع في ملف مؤقت لمعالجته
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.info("جاري تحليل الفيديو وعرض النتائج..." if is_ar else "Analyzing video...")
        st_frame = st.empty()
        
        counter = 0
        stage = None

        with PoseLandmarker.create_from_options(options) as landmarker:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                
                detection_result = landmarker.detect(mp_image)

                is_correct = True
                angle = 0
                feedback = "Ready"

                if detection_result.pose_landmarks:
                    for landmarks in detection_result.pose_landmarks:
                        try:
                            is_correct, angle, feedback = process_exercise(exercise, landmarks)
                        except Exception:
                            pass

                        if exercise in ["Squat", "Push-up"]:
                            if angle > 150:
                                stage = "up"
                            if angle < 90 and stage == "up":
                                stage = "down"
                                counter += 1

                # تصميم المربع والتنبيهات الملونة على الفيديو
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                
                overlay = rgb.copy()
                cv2.rectangle(overlay, (15, 15), (460, 130), (15, 23, 42), -1)
                cv2.addWeighted(overlay, 0.75, rgb, 0.25, 0, rgb)
                
                border_color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.rectangle(rgb, (0, 0), (rgb.shape[1], rgb.shape[0]), border_color, 4)

                status_str = "CORRECT / تمرير صحيح" if is_correct else "INCORRECT / عدل وضعيتك"
                cv2.putText(rgb, f"Status: {status_str}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.putText(rgb, f"Count: {counter}  |  Angle: {int(angle)}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(rgb, f"Tip: {feedback}", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

                st_frame.image(rgb, channels="RGB", use_container_width=True)

        cap.release()
        st.success("تم إتمام تحليل الفيديو بنجاح! 🎉" if is_ar else "Analysis complete!")
else:
    st.warning("الرجاء رفع فيديو تمرين من القائمة أعلاه للبدء بالعرض." if is_ar else "Please upload a video to start.")
