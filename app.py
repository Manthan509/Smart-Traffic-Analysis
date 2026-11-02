
import os
import tempfile
import subprocess
import io
from pathlib import Path

import cv2
import streamlit as st
import pandas as pd
import numpy as np
import imageio_ffmpeg

from PIL import Image
from ultralytics import YOLO


st.set_page_config(
    page_title="Smart Traffic Analysis",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: #E5E7EB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263449;
}

section[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p, label, span {
    color: #CBD5E1;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #17243A, #111C30);
    border: 1px solid #263D5B;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

div[data-testid="stMetricValue"] {
    color: #60A5FA !important;
}

.stButton > button {
    background: linear-gradient(90deg, #2563EB, #0EA5E9);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1D4ED8, #0284C7);
    border: none;
    color: white;
}

div[data-testid="stFileUploader"] {
    background-color: #111C30;
    border: 1px dashed #3B82F6;
    border-radius: 12px;
    padding: 15px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] {
    background-color: #17243A;
    border-color: #334155;
}

button[data-baseweb="tab"] {
    color: #94A3B8;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #60A5FA;
}

.custom-card {
    background-color: #111C30;
    border: 1px solid #263D5B;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

.status-box {
    background-color: #102D2A;
    border: 1px solid #166534;
    border-radius: 10px;
    padding: 12px;
    color: #4ADE80;
}

.footer {
    text-align: center;
    color: #64748B;
    padding-top: 30px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


MODEL_PATH = Path(__file__).parent / "yolov8n.pt"

VEHICLE_CLASSES = {
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7
}

PERSON_CLASS = 0


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


def get_counts(result):
    counts = {
        "car": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0,
        "bicycle": 0,
        "person": 0
    }

    if result.boxes is None:
        return counts

    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id == PERSON_CLASS:
            counts["person"] += 1

        for vehicle, vehicle_id in VEHICLE_CLASSES.items():
            if class_id == vehicle_id:
                counts[vehicle] += 1

    return counts


def calculate_density(vehicle_count):
    if vehicle_count <= 5:
        return "Low"
    elif vehicle_count <= 15:
        return "Medium"
    return "High"


def create_statistics(counts):
    return pd.DataFrame({
        "Vehicle Type": [
            "Car",
            "Bus",
            "Truck",
            "Motorcycle",
            "Bicycle"
        ],
        "Count": [
            counts["car"],
            counts["bus"],
            counts["truck"],
            counts["motorcycle"],
            counts["bicycle"]
        ]
    })


def show_density(density):
    if density == "Low":
        color = "#22C55E"
        emoji = "🟢"
    elif density == "Medium":
        color = "#F59E0B"
        emoji = "🟡"
    else:
        color = "#EF4444"
        emoji = "🔴"

    st.markdown(
        f"""
        <div style="
            background-color:#17243A;
            border:1px solid {color};
            border-radius:12px;
            padding:18px;
            text-align:center;
        ">
            <div style="color:#94A3B8;font-size:14px;">
                TRAFFIC DENSITY
            </div>
            <div style="
                color:{color};
                font-size:28px;
                font-weight:700;
                margin-top:8px;
            ">
                {emoji} {density}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def process_video(video_path, model, confidence):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions.")

    raw_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )
    raw_path = raw_file.name
    raw_file.close()

    final_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )
    final_path = final_file.name
    final_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        raw_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        os.remove(raw_path)
        os.remove(final_path)
        raise ValueError("Unable to create output video.")

    total_frames = 0
    total_vehicles = 0
    total_people = 0

    vehicle_totals = {
        "car": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0,
        "bicycle": 0
    }

    density_distribution = {
        "Low": 0,
        "Medium": 0,
        "High": 0
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(
                frame,
                conf=confidence,
                verbose=False
            )

            result = results[0]
            annotated_frame = result.plot()
            counts = get_counts(result)

            frame_vehicles = sum(
                counts[name] for name in VEHICLE_CLASSES
            )

            total_vehicles += frame_vehicles
            total_people += counts["person"]

            for vehicle in VEHICLE_CLASSES:
                vehicle_totals[vehicle] += counts[vehicle]

            frame_density = calculate_density(frame_vehicles)
            density_distribution[frame_density] += 1

            out.write(annotated_frame)
            total_frames += 1

            if total_video_frames > 0:
                progress = min(
                    total_frames / total_video_frames,
                    1.0
                )
                progress_bar.progress(progress)

            status_text.text(
                f"Processing frame {total_frames} "
                f"of {total_video_frames or '?'}"
            )

    finally:
        cap.release()
        out.release()

    if total_frames == 0:
        os.remove(raw_path)
        os.remove(final_path)
        raise ValueError("No frames could be processed.")

    progress_bar.progress(1.0)
    status_text.success("Video processing completed!")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_exe,
        "-y",
        "-i", raw_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        final_path
    ]

    conversion = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if conversion.returncode != 0:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists(final_path):
            os.remove(final_path)

        raise RuntimeError(
            "Video conversion failed: " + conversion.stderr[-1500:]
        )

    if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
        raise RuntimeError("Converted video is empty.")

    os.remove(raw_path)

    average_vehicles = total_vehicles / total_frames
    average_people = total_people / total_frames

    return {
        "video_path": final_path,
        "frames": total_frames,
        "vehicles": total_vehicles,
        "people": total_people,
        "vehicle_totals": vehicle_totals,
        "density_distribution": density_distribution,
        "average_vehicles": average_vehicles,
        "average_people": average_people
    }


with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding:15px 0;">
            <div style="font-size:42px;">🚦</div>
            <h2 style="margin:0;color:#60A5FA;">
                TRAFFIC AI
            </h2>
            <p style="font-size:12px;color:#94A3B8;">
                Smart Traffic Monitoring System
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Image Analysis", "Video Analysis"],
        index=0
    )

    st.divider()

    st.markdown("### ⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05
    )

    st.caption(
        "Higher confidence values filter out less certain detections."
    )

    st.divider()

    st.markdown(
        """
        <div class="status-box">
            🟢 Application Ready
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="padding-top:20px;color:#64748B;font-size:12px;">
            Powered by YOLOv8 and OpenCV
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    "<p style='color:#60A5FA; font-size:13px; "
    "font-weight:600; letter-spacing:2px;'>"
    "AI-POWERED COMPUTER VISION</p>",
    unsafe_allow_html=True
)

st.title("🚦 Smart Traffic Analysis")

st.caption(
    "Intelligent vehicle detection, traffic monitoring "
    "and density analysis."
)

st.divider()


try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


if page == "Image Analysis":

    st.markdown("## 🖼️ Image Analysis")

    st.caption(
        "Upload a traffic image to detect and analyse vehicles and people."
    )

    uploaded_image = st.file_uploader(
        "Upload Traffic Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")

        st.markdown("### Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button(
            "🚀 Analyse Image",
            use_container_width=True
        ):

            with st.spinner("Running YOLO object detection..."):

                image_array = cv2.cvtColor(
                    np.array(image),
                    cv2.COLOR_RGB2BGR
                )

                results = model.predict(
                    image_array,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]
                counts = get_counts(result)

                total_vehicles = sum(
                    counts[name] for name in VEHICLE_CLASSES
                )

                total_people = counts["person"]
                density = calculate_density(total_vehicles)

                annotated_image = result.plot()

                annotated_image = cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                )

            st.success("Image analysis completed!")

            st.markdown("## 📊 Analysis Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Vehicles", total_vehicles)

            with col2:
                st.metric("People Detected", total_people)

            with col3:
                st.metric(
                    "Total Objects",
                    total_vehicles + total_people
                )

            with col4:
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.0f}%"
                )

            st.markdown("### 🚦 Traffic Density")

            show_density(density)

            st.markdown("## 🔍 Detection Results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("### YOLO Detection")
                st.image(
                    annotated_image,
                    use_container_width=True
                )

            st.markdown("## 🚘 Vehicle Statistics")

            statistics = create_statistics(counts)

            col1, col2 = st.columns([1, 1.3])

            with col1:
                st.dataframe(
                    statistics,
                    hide_index=True,
                    use_container_width=True
                )

            with col2:
                chart_data = statistics.set_index("Vehicle Type")

                st.bar_chart(
                    chart_data,
                    color="#3B82F6"
                )

            image_buffer = io.BytesIO()

            Image.fromarray(annotated_image).save(
                image_buffer,
                format="PNG"
            )

            st.download_button(
                "⬇️ Download Analysed Image",
                data=image_buffer.getvalue(),
                file_name="traffic_analysis.png",
                mime="image/png",
                use_container_width=True
            )


elif page == "Video Analysis":

    st.markdown("## 🎥 Video Analysis")

    st.caption(
        "Upload a traffic video to perform frame-by-frame object detection."
    )

    uploaded_video = st.file_uploader(
        "Upload Traffic Video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button(
            "🚀 Analyse Video",
            use_container_width=True
        ):

            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_video.name).suffix,
                delete=False
            ) as temp_video:

                temp_video.write(uploaded_video.getbuffer())
                input_path = temp_video.name

            try:

                with st.spinner(
                    "Processing video frames using YOLO..."
                ):

                    analysis = process_video(
                        input_path,
                        model,
                        confidence
                    )

                st.success("Video analysis completed successfully!")

                st.markdown("## 📊 Video Analysis Overview")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Frames",
                        analysis["frames"]
                    )

                with col2:
                    st.metric(
                        "Vehicle Detections",
                        analysis["vehicles"]
                    )

                with col3:
                    st.metric(
                        "People Detections",
                        analysis["people"]
                    )

                with col4:
                    st.metric(
                        "Avg Vehicles / Frame",
                        f"{analysis['average_vehicles']:.2f}"
                    )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("### 🚘 Vehicle Distribution")

                    vehicle_data = pd.DataFrame({
                        "Vehicle Type": [
                            "Car",
                            "Bus",
                            "Truck",
                            "Motorcycle",
                            "Bicycle"
                        ],
                        "Count": [
                            analysis["vehicle_totals"]["car"],
                            analysis["vehicle_totals"]["bus"],
                            analysis["vehicle_totals"]["truck"],
                            analysis["vehicle_totals"]["motorcycle"],
                            analysis["vehicle_totals"]["bicycle"]
                        ]
                    })

                    st.dataframe(
                        vehicle_data,
                        hide_index=True,
                        use_container_width=True
                    )

                    st.bar_chart(
                        vehicle_data.set_index("Vehicle Type"),
                        color="#3B82F6"
                    )

                with col2:

                    st.markdown("### 🚦 Traffic Density Distribution")

                    density_data = pd.DataFrame({
                        "Density": ["Low", "Medium", "High"],
                        "Frames": [
                            analysis["density_distribution"]["Low"],
                            analysis["density_distribution"]["Medium"],
                            analysis["density_distribution"]["High"]
                        ]
                    })

                    st.bar_chart(
                        density_data.set_index("Density"),
                        color="#0EA5E9"
                    )

                    st.markdown("### 👥 People Analysis")

                    st.metric(
                        "Average People Per Frame",
                        f"{analysis['average_people']:.2f}"
                    )

                st.markdown("## 🎬 Processed Traffic Video")

                st.video(analysis["video_path"])

                with open(
                    analysis["video_path"],
                    "rb"
                ) as video_file:

                    st.download_button(
                        "⬇️ Download Analysed Video",
                        data=video_file.read(),
                        file_name="analysed_traffic.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Video processing error: {e}")

            finally:
                if os.path.exists(input_path):
                    os.remove(input_path)


st.markdown(
    """
    <div class="footer">
        SMART TRAFFIC ANALYSIS SYSTEM
        <br>
        Powered by YOLOv8 | Python | OpenCV | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
