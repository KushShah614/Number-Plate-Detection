import streamlit as st
import cv2  # used to bring in the OpenCV library in Python.
# for images and videos

from ultralytics import YOLO
import numpy as np
from PIL import Image # use for opening, editing, and saving images in Python.
import os


@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

#  Streamlit UI
st.title("Number Plate Detection")

uploaded_file = st.file_uploader(
    "Upload an Image or Video File",
    type=["jpg", "jpeg", "png", "mp4", "mkv"]
)


os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=320)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f'{confidence*100:.2f}%',
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    2
                )

        out.write(frame)

    cap.release()
    out.release()

    return output_path



def process_media(input_path, output_path):
    file_extension = os.path.splitext(input_path)[1].lower()

    if file_extension in ['.mp4', '.mkv']:
        return process_video(input_path, output_path)

    elif file_extension in ['.jpg', '.jpeg', '.png']:
        return predict_and_save_image(input_path, output_path)

    else:
        st.error(f"Unsupported File Type: {file_extension}")
        return None

# Prediction Code
def predict_and_save_image(path_test_car, output_image_path):
    results = model.predict(path_test_car, imgsz=320)

    image = cv2.imread(path_test_car)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put confidence text
            cv2.putText(
                image,
                f'{confidence*100:.2f}%',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    # Save output image
    cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return output_image_path


# logic
if uploaded_file is not None:
    input_path = f"temp/{uploaded_file.name}"
    output_path = f"output/{uploaded_file.name}"

    # Save uploaded file
    with open(input_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing..."):
        result_path = process_media(input_path, output_path)

    # Display result
    if result_path:
        st.success("Detection Complete!")

        if input_path.endswith(('.mp4', '.mkv')):
            st.video(result_path)
        else:
            st.image(Image.open(result_path), caption="Detected Image", width='stretch')