import time
import cv2
import torch
import os
import io
import streamlit as st
from ultralytics import YOLO

def upload_model():
    """Uploads the model file and saves it to the current working directory."""
    st.title("Upload Your YOLO Model")
    model_file = st.file_uploader("Upload Model File (.pt)", type=["pt"])
    if model_file is not None:
        model_path = os.path.join(os.getcwd(), model_file.name)
        with open(model_path, "wb") as out:  # Save uploaded model file
            out.write(model_file.read())
        st.success(f"Model file '{model_file.name}' uploaded successfully!")

def inference():
    # Hide main menu style
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""
    # Main title of streamlit application
    main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px;
                             font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                             Welcome To The Onbiron Object Detection Page 
                    </h1></div>"""
    # Subtitle of streamlit application
    sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center;
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;"> </h4>
                    </div>"""
    # Append the custom HTML
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title_cfg, unsafe_allow_html=True)

    # Initialize session state for selected classes if not already present
    if 'selected_classes' not in st.session_state:
        st.session_state.selected_classes = []

    # Add elements to vertical setting menu
    st.sidebar.title("User Configuration")
    # Add video source selection dropdown
    source = st.sidebar.selectbox(
        "Video",
        ("webcam", "video"),
    )

    vid_file_name = ""
    if source == "video":
        vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg"])
        if vid_file is not None:
            g = io.BytesIO(vid_file.read())  # BytesIO Object
            vid_location = "ultralytics.mp4"
            with open(vid_location, "wb") as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file
            vid_file_name = "ultralytics.mp4"
    elif source == "webcam":
        vid_file_name = 0

    # Add dropdown menu for model selection
    available_models = [file.replace(".pt", "") for file in os.listdir() if file.endswith(".pt")]
    selected_model = st.sidebar.selectbox("Model", available_models)

    if selected_model:
        with st.spinner("Model is loading..."):
            model = YOLO(f"{selected_model}.pt")  # Load the YOLO model
            class_names = list(model.names.values())  # Convert dictionary to list of class names
        st.success("Model loaded successfully!")

        # Create buttons for Select All and Clear All
        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            if st.button("Select All Classes"):
                st.session_state.selected_classes = class_names
        with col2:
            if st.button("Clear All Classes"):
                st.session_state.selected_classes = []

        # Multiselect box for classes
        selected_classes = st.sidebar.multiselect("Classes", class_names, default=st.session_state.selected_classes)

        # Update session state
        st.session_state.selected_classes = selected_classes

        # Update selected indices
        selected_ind = [class_names.index(option) for option in selected_classes]
        if not isinstance(selected_ind, list):  # Ensure selected_ind is a list
            selected_ind = list(selected_ind)

        conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01))
        iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))

        col1 = st.columns(1)[0]  # Display only one column for the annotated frame
        ann_frame = col1.empty()
        fps_display = st.sidebar.empty()  # Placeholder for FPS display

        if st.sidebar.button("Start"):
            videocapture = cv2.VideoCapture(vid_file_name)  # Capture the video
            if not videocapture.isOpened():
                st.error("Could not open webcam.")

            stop_button = st.button("Stop")  # Button to stop the inference

            while videocapture.isOpened():
                success, frame = videocapture.read()
                if not success:
                    st.warning("Failed to read frame from webcam. Please make sure the webcam is connected properly.")
                    break

                prev_time = time.time()
                # Store model predictions
                results = model.track(frame, conf=conf, iou=iou, classes=selected_ind, persist=True)

                # Remove object IDs from labels
                annotated_frame = frame.copy()
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Box coordinates
                    scores = result.boxes.conf.cpu().numpy()  # Confidence scores
                    cls_inds = result.boxes.cls.cpu().numpy()  # Class indices
                    for box, score, cls_ind in zip(boxes, scores, cls_inds):
                        label = f"{class_names[int(cls_ind)]} {score:.2f}"
                        (x1, y1, x2, y2) = map(int, box)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Calculate model FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                # Display annotated frame
                ann_frame.image(annotated_frame, channels="BGR")
                if stop_button:
                    videocapture.release()  # Release the capture
                    torch.cuda.empty_cache()  # Clear CUDA memory
                    st.stop()  # Stop streamlit app
                # Display FPS in sidebar
                fps_display.metric("FPS", f"{fps:.2f}")

            # Release the capture
            videocapture.release()

        # Clear CUDA memory
        torch.cuda.empty_cache()
        # Destroy window
        cv2.destroyAllWindows()
    else:
        st.warning("Please upload and select a model file.")

# Main function call
if __name__ == "__main__":
    st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide", initial_sidebar_state="auto")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Upload Model", "Run Inference"])
    if app_mode == "Upload Model":
        upload_model()
    elif app_mode == "Run Inference":
        inference()
