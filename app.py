from flask import Flask, render_template, request, redirect, url_for, Response
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker
from easydict import EasyDict as edict
import cv2
import numpy as np
import torch
import time
import ultralytics 
import torch
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')

def webcam_generator():
    #Initialize person count
    person_count = 0

    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Load model
    model = ultralytics.YOLO("yolo-Weights/yolov8n.pt")

    # Object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    # Threshold for confidence
    confidence_threshold = 0.5

    while True:

        success, img = cap.read()
        if not success:
            break

        person_count = 0
        results = model(img, stream=True)

        

        # Process detection results
        for r in results:
            boxes = r.boxes

            for box in boxes:
                person_count += 1
                
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # Confidence level
                confidence = box.conf[0]

                # Class index
                cls = int(box.cls[0])

                # If the detected object is a person and confidence level is above the threshold
                if classNames[cls] == "person" and confidence >= confidence_threshold:

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Display confidence
                    print("Confidence --->", confidence)

                    # Display class name
                    print("Class name -->", classNames[cls])

                    # Object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                    
        
        # Convert the frame to JPEG format
        _, jpeg_frame = cv2.imencode('.jpg', img)
        frame_bytes = jpeg_frame.tobytes()
        cv2.putText(img, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        _, jpeg_frame = cv2.imencode('.jpg', img)
        frame_bytes = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Release the capture
    cap.release()


@app.route('/webcam_feed')
def webcam_feed():
    return Response(webcam_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


    
@app.route('/process_video', methods=['POST'])
def process_video():
    # Get the uploaded file
    video_file = request.files['video']

    # Save the video file
    video_path = 'uploaded_video.mp4'
    video_file.save(video_path)

    # DeepSORT setup
    deep_sort_weights = './deep_sort/deep/checkpoint/ckpt.t7'
    tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = './output.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    device = torch.device('cpu')

    frames = []

    unique_track_ids = set()

    counter, fps, elapsed = 0, 0, 0
    start_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = og_frame.copy()

            model = ultralytics.YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
            results = model(frame, device='cpu', classes=0, conf=0.8)

            class_names = ['person']
    
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                probs = result.probs  # Class probabilities for classification outputs
                cls = boxes.cls.tolist()  # Convert tensor to list
                xyxy = boxes.xyxy
                conf = boxes.conf
                xywh = boxes.xywh  # box with xywh format, (N, 4)
                for class_index in cls:
                    class_name = class_names[int(class_index)]
                    #print("Class:", class_name)

            pred_cls = np.array(cls)
            conf = conf.detach().cpu().numpy()
            xyxy = xyxy.detach().cpu().numpy()
            bboxes_xywh = xywh
            bboxes_xywh = xywh.cpu().numpy()
            bboxes_xywh = np.array(bboxes_xywh, dtype=float)

            tracks = tracker.update(bboxes_xywh, conf, og_frame)

            for track in tracker.tracker.tracks:
                track_id = track.track_id
                hits = track.hits
                x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
                w = x2 - x1  # Calculate width
                h = y2 - y1  # Calculate height

                # Set color values for red, blue, and green
                red_color = (0, 0, 255)  # (B, G, R)
                blue_color = (255, 0, 0)  # (B, G, R)
                green_color = (0, 255, 0)  # (B, G, R)

                # Determine color based on track_id
                color_id = track_id % 3
                if color_id == 0:
                    color = red_color
                elif color_id == 1:
                    color = blue_color
                else:
                    color = green_color

                cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

                text_color = (0, 0, 0)  # Black color for text
                cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

                # Add the track_id to the set of unique track IDs
                unique_track_ids.add(track_id)

            # Update the person count based on the number of unique track IDs
            person_count = len(unique_track_ids)
            person_count = person_count / 2

            # Update FPS and place on frame
            current_time = time.perf_counter()
            elapsed = (current_time - start_time)
            counter += 1
            if elapsed > 1:
                fps = counter / elapsed
                counter = 0
                start_time = current_time

            # Draw person count on frame
            cv2.putText(og_frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Append the frame to the list
            frames.append(og_frame)

            # Write the frame to the output video file
            out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

            #Show the frame
            cv2.imshow("Video", og_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
    
    

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)

