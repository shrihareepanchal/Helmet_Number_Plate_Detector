import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
import os
from datetime import datetime
import xlwings as xw
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # To ignore any warning


# Initializing PaddleOCR for Recognition of License Plate
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Image preprocessing function for OCR Results
def preprocess_image(image):
    # Convert input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    # Applying Adaptive Thresholding to convert image to binary using Gaussian sum
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)  # Applying for Morphological Operation to make characters more visible
    # Text characters made thicker and readable
    dilation = cv2.dilate(threshold, kernel, iterations=1)
    return dilation

# Function for Performing OCR on an Image Array to extract text
def perform_ocr(image_array):
    if image_array is None or image_array.size == 0:
        return None

    # Ensure image has 3 channels
    if len(image_array.shape) == 2 or image_array.shape[2] == 1:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

    # OCR operation
    results = ocr.ocr(image_array)
    detected_text = []

    if results and results[0]:
        for line in results[0]:
            if line and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                text, conf = line[1][0], float(line[1][1])
                if conf > 0.5:
                    detected_text.append(text)

    return ' '.join(detected_text) if detected_text else None


# Function for callback for RGB Window
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Triggers when mouse over window
        point = [x, y]
        print(point)
        
cv2.namedWindow('RGB')  # RGB Window
cv2.setMouseCallback('RGB', RGB)  # RGB function for RGB window

# Loading YOLOv8 Model
model = YOLO("best.pt") # Load with weights_only=False if trusted
model.to("cpu")  # Model running on CPU
names = model.names
print(names)

# Polygon area
area = [(50, 100), (50, 400), (400, 400), (400, 100)]  # Left lane

# Create directory for current date
curr_date = datetime.now().strftime('%d-%m-%y')
if not os.path.exists(curr_date):
    os.makedirs(curr_date)
    
# Path for Excel file in curr_date directory
excel_file_path = os.path.join(curr_date, f'{curr_date}.xlsx')

# Open Excel file with xlwings
try:
    wb = xw.Book(excel_file_path) if os.path.exists(excel_file_path) else xw.Book()
except Exception as e:
    print(f"Error opening Excel file: {e}")
    wb = xw.Book()
ws = wb.sheets[0]
if ws.range('A1').value is None:
    ws.range('A1').value = ['Number Plate', 'Date', 'Time']  # Write Heading

# Track Processed data IDs
processed_track_ids = set()

# Opening video file
cap = cv2.VideoCapture('sample.mp4')
if not cap.isOpened():
    print("Error: Could not open video file 'sample.mp4'.")
    exit()

# Process video frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 Model to track the Frame
    results = model.track(frame, persist=True)
    
    # Initialize flags and Variables
    no_helmet_detected = False
    numberplate_box = None
    numberplate_track_id = None
    
    # Check if YOLO detected any objects
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Extract boxes, class IDs, track IDs, confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding Coordinates
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence Scores
    
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2  # Center point at x
            cy = (y1 + y2) // 2  # Center point at y
            
            result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
            if result >= 0:
                if c == 'no-helmet':
                    no_helmet_detected = True
                elif c == 'numberplate':
                    numberplate_box = box  # Store the numberplate bounding box
                    numberplate_track_id = track_id
    
    # If no-helmet and numberplate are detected and the track ID is not already processed
    if no_helmet_detected and numberplate_box is not None and numberplate_track_id not in processed_track_ids:
        x1, y1, x2, y2 = numberplate_box
        # Cropping area slightly to capture the whole plate
        crop = frame[max(0, y1-10):min(frame.shape[0], y2+10), max(0, x1-10):min(frame.shape[1], x2+10)]
    
        # Skip processing if image is empty or too small
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            print("Cropped Image is too Small so Skip Processing.")
            continue
    
        crop = cv2.resize(crop, (320, 85))  # Resizes the cropped number plate image
        cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)

        # Debug: Save and display the cropped image
        cv2.imwrite("debug_crop.jpg", crop)
        cv2.imshow("Debug Crop", crop)
        cv2.waitKey(0)  # Pause to inspect
    
        # Performing OCR Operation on cropped Number plate image
        text = perform_ocr(crop)
        print(f'Detected Number Plate: {text}')
    
        # Check OCR Text
        if not text:
            print("OCR failed to detect a valid Number Plate.")
            # Saving the failed image
            curr_time = datetime.now().strftime('%H-%M-%S-%f')[:12]
            failed_image_path = os.path.join(curr_date, f"failed_{curr_time}.jpg")
            cv2.imwrite(failed_image_path, crop)
            continue  # Skips to the Next Frame
    
        # Save cropped image with current time
        curr_time = datetime.now().strftime('%H-%M-%S-%f')[:12]
        crop_image_path = os.path.join(curr_date, f'{text}_{curr_time}.jpg')
        cv2.imwrite(crop_image_path, crop)
    
        # Save data to Excel
        last_row = ws.range('A' + str(ws.cells.last_cell.row)).end('up').row
        print(f"Saving to Excel: {text}, {curr_date}, {curr_time}")
        ws.range(f"A{last_row+1}").value = [text, curr_date, curr_time]
    
        # Adds the number plate track ID to processed_track_ids
        processed_track_ids.add(numberplate_track_id)

    # Draws the polygon area on the frame
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)
    
    # Display the processed frame in the RGB window
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()  # Closes all windows

# Saves the Excel file to path
try:
    wb.save(excel_file_path)
    wb.app.quit()  # Ensure Excel instance is properly closed
except Exception as e:
    print(f"Error saving or closing Excel file: {e}")