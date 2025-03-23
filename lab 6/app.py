import cv2
import numpy as np
import torch
import folium
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load YOLOv8 model for animal detection
model = YOLO("yolov8n.pt")  # You can replace this with a custom-trained model

# Open webcam (or replace 0 with a video file path)
cap = cv2.VideoCapture(0)

# Store detected locations
alert_locations = []
base_lat, base_lon = 31.5204, 74.3587  # Base location (Lahore, Pakistan)

# Function to detect animals and generate video frames
def generate_frames():
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # Stop if the frame is not read properly
        
        # Run YOLO detection
        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0].item())]
                
                if conf > 0.5:  # Confidence threshold
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Simulate GPS coordinates (Modify for real GPS data)
                    lat = base_lat + np.random.uniform(-0.01, 0.01)
                    lon = base_lon + np.random.uniform(-0.01, 0.01)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Store detection data
                    alert_locations.append((lat, lon, label, timestamp))
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        frame_count += 1
        if frame_count >= 100:  # Stop after 100 frames (for testing)
            break

# Route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to display home page with video and map
@app.route('/')
def index():
    return render_template('index.html')

# Route to generate and save the map with detected animals
@app.route('/stop')
def stop():
    cap.release()
    
    # Create a Folium map centered at the base location
    m = folium.Map(location=(base_lat, base_lon), zoom_start=12)

    # Add markers for detected animals
    for lat, lon, label, timestamp in alert_locations:
        popup_text = f"{label} detected at {timestamp}"
        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color="red", icon="paw")
        ).add_to(m)

    # Save the map
    map_path = "static/herd_alert_map.html"
    m.save(map_path)
    return f"Detection stopped. <br> <a href='{map_path}' target='_blank'>View Animal Detection Map</a>"

if __name__ == '__main__':
    app.run(debug=True)
