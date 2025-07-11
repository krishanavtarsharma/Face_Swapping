import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector

st.set_page_config(page_title="Face Swapper", layout="centered")
st.title("🤳 Real-Time Face Swapper using Webcam (No Lag ✅)")

detector = FaceDetector()

# Session variables
if 'face1' not in st.session_state:
    st.session_state.face1 = None
if 'face2' not in st.session_state:
    st.session_state.face2 = None

# Custom video frame processor
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        self.frame = img
        return img

# Start webcam with webrtc
ctx = webrtc_streamer(
    key="face-swapper",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False}
)

if ctx.video_transformer:
    st.markdown("### 📸 Capture & Swap")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📷 Capture Face 1"):
            st.session_state.face1 = ctx.video_transformer.frame
            st.success("✅ Face 1 Captured")

    with col2:
        if st.button("📷 Capture Face 2"):
            st.session_state.face2 = ctx.video_transformer.frame
            st.success("✅ Face 2 Captured")

    with col3:
        if st.button("🔁 Swap Faces"):
            face1 = st.session_state.face1
            face2 = st.session_state.face2

            if face1 is not None and face2 is not None:
                face1 = cv2.resize(face1, (640, 480))
                face2 = cv2.resize(face2, (640, 480))

                face1_detected, bboxs1 = detector.findFaces(face1)
                face2_detected, bboxs2 = detector.findFaces(face2)

                if bboxs1 and bboxs2:
                    x1, y1, w1, h1 = bboxs1[0]['bbox']
                    x2, y2, w2, h2 = bboxs2[0]['bbox']

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    w1, h1, w2, h2 = map(int, [w1, h1, w2, h2])

                    crop1 = face1[y1:y1+h1, x1:x1+w1]
                    crop2 = face2[y2:y2+h2, x2:x2+w2]

                    target_w = min(w1, w2)
                    target_h = min(h1, h2)

                    crop1_resized = cv2.resize(crop1, (target_w, target_h))
                    crop2_resized = cv2.resize(crop2, (target_w, target_h))

                    face1[y1:y1+target_h, x1:x1+target_w] = crop2_resized
                    face2[y2:y2+target_h, x2:x2+target_w] = crop1_resized

                    st.image(face1, caption="Swapped Face 1", channels="BGR")
                    st.image(face2, caption="Swapped Face 2", channels="BGR")
                else:
                    st.warning("❌ Face not detected in one or both faces.")
            else:
                st.warning("⚠️ Please capture both faces before swapping.")
