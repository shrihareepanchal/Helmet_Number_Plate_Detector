Helmet & Number Plate Detection Project
Overview
This project implements an automated system for detecting helmets and number plates in video streams using computer vision techniques. It utilizes the YOLOv8 model for object detection and tracking, PaddleOCR for optical character recognition (OCR) of number plates, and Streamlit for creating an interactive dashboard to visualize the results.
Features

Detects "no-helmet" and "number plate" objects in real-time video.
Tracks detected objects across frames using YOLOv8.
Performs OCR on cropped number plate images to extract text.
Logs detected number plates with timestamps in an Excel file.
Provides a Streamlit dashboard to view logged data and captured images.

Requirements

Python 3.x
Required libraries:
opencv-python (cv2)
ultralytics (for YOLOv8)
cvzone
paddleocr
os
datetime
xlwings
torch
numpy
streamlit
pandas



Install dependencies using:
pip install opencv-python ultralytics cvzone paddleocr os-sys datetime xlwings torch numpy streamlit pandas

Setup

Clone the Repository:
git clone <repository-url>
cd Helmate_Number_plate_Detection


Prepare the Environment:

Create a virtual environment (optional):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required libraries as listed above.



Prepare Model and Video:

Place the YOLOv8 model weights file (best.pt) in the project directory.
Ensure a sample video file (sample.mp4) is available in the project directory.


Run the Application:

Start the detection script:
python main.py


Launch the Streamlit dashboard:
streamlit run streamlit_app.py





Usage

Main Script (main.py):

Processes the sample.mp4 video, detects helmets and number plates, and saves results to an Excel file and image directory based on the current date.
A debug window ("RGB") displays the processed video frames.


Dashboard (streamlit_app.py):

Enter the "Log Folder Date" in the sidebar (e.g., "02-07-25").
View the loaded Excel log data and recently captured plate images if available.
Download the Excel file using the "Download Excel" button.



File Structure

main.py: Core script for video processing, object detection, and OCR.
streamlit_app.py: Streamlit application for the dashboard.
best.pt: Pre-trained YOLOv8 model weights.
sample.mp4: Sample video file for testing.

Troubleshooting

OCR Fails: If number plates are not detected, check the cropped image quality (saved as debug_crop.jpg during debug mode). Adjust preprocessing or cropping parameters.
No Data in Dashboard: Ensure the date folder matches the log folder date and that main.py has generated the Excel file and images.
Video Errors: Verify sample.mp4 is accessible and playable.

Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the project.
License
This project is licensed under the MIT License - see the LICENSE file for details.
