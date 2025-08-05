# Player-Re-identification
Player Re-Identification in Sports Video Using YOLOv8 + DeepSORT Objective

This project addresses a real-world sports analytics problem: tracking and re-identifying football players in a video. Each player must retain a unique and consistent ID, even after leaving or re-entering the frame. The solution uses YOLOv8 for object detection and DeepSORT for multi-object tracking, with a custom ID assignment strategy based on IoU-based matching for high stability.

Project Structure

Assignment/ │ ├── best.pt # Trained YOLOv8 model (detects players & ball) ├── 15sec_input_720p.mp4 # 15-second input video (720p resolution) ├── Result/ │ └── final_output.mp4 # Output video with stable player tracking │ ├── player_tracking_final.ipynb # Main notebook with final working code ├── README.md # Setup & documentation (this file) ├── report.md

How to Set Up and Run

Step 1: Mount Google Drive (in Colab)

python

from google.colab import drive drive.mount('/content/drive')

Step 2: Install Required Libraries bash Copy Edit !pip install ultralytics !pip install deep_sort_realtime !pip install opencv-python-headless ✅ Step 3: Run the Main Code Open player_tracking_final.ipynb and run all cells sequentially.

Ensure these paths are correct:

python

model_path = '/content/drive/MyDrive/Assignment Materials/Assignment Materials/best.pt' input_path = '/content/drive/MyDrive/Assignment Materials/Assignment Materials/15sec_input_720p.mp4' output_path = '/content/drive/MyDrive/Assignment Materials/Assignment Materials/Result/final_output.mp4'

Dependencies / Environment Python 3.8+ Google Colab or local Jupyter Notebook ultralytics (for YOLOv8) deep_sort_realtime (for tracking) opencv-python-headless matplotlib, numpy

To install all: pip install ultralytics deep_sort_realtime opencv-python-headless matplotlib numpy
Methodology Overview

Detection with YOLOv8

A custom-trained model detects players (class ID 2) and filters out others.

Only confident detections (confidence > 0.6) are passed to tracking.

Tracking with DeepSORT

DeepSORT assigns short-term track IDs using motion and appearance embeddings.

Global Player ID Assignment

A global player_id is assigned once, using IoU (Intersection-over-Union) between new and past bounding boxes.

Even if DeepSORT ID resets, the same player keeps their global ID.

ID Stability Enhancements

max_age = 40 to retain tracks when players go off-frame.

n_init = 8 ensures only stable tracks are counted.

No bounding boxes are shown unless a player is confidently detected.

Output The final_output.mp4 shows each detected player with:

A green bounding box

A unique and persistent ID (Player 1, Player 2, etc.)

IDs remain consistent through motion, occlusion, and reappearance.

Accuracy and Reliability Fully stable tracking for individual players

No duplicate IDs

Robust even in crowded frames

No extra bounding boxes without players
