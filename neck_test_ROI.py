import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

def select_neck_roi(frame):
    r = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return r

def save_roi_dimensions(roi, file_path):
    roi_data = {
        "x": roi[0],
        "y": roi[1],
        "width": roi[2],
        "height": roi[3]
    }
    with open(file_path, 'w') as file:
        json.dump(roi_data, file)

def load_roi_dimensions(file_path):
    with open(file_path, 'r') as file:
        roi_data = json.load(file)
    return (roi_data['x'], roi_data['y'], roi_data['width'], roi_data['height'])

def calculate_ppw_signal(video_path, roi):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    x, y, w, h = roi
    ppw_signal = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        neck_roi = frame[y:y+h, x:x+w]
        gray_frame = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)

        # Calculate the mean intensity within the ROI
        mean_intensity = np.mean(gray_frame)
        ppw_signal.append(mean_intensity)

    cap.release()
    return ppw_signal

def plot_ppw_signal(ppw_signal):
    plt.figure(figsize=(10, 4))
    plt.plot(ppw_signal, label='PPW Signal')
    plt.xlabel('Frame')
    plt.ylabel('Mean Intensity')
    plt.title('Neck-PPW Signal')
    plt.legend()
    plt.show()

def main():
    dataset_folder = "/Users/danillugli/Desktop/Boccignone/BP4D+"
    subject = "F002"
    task = "T3/"
    video_path = dataset_folder + f"/{subject}/{task}vid.avi"
    roi_file_path = dataset_folder + f"/{subject}/{task}roi_dimensions.json"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    # Select ROI and save dimensions
    roi = select_neck_roi(frame)
    save_roi_dimensions(roi, roi_file_path)
    
    # Calculate and plot the PPW signal
    ppw_signal = calculate_ppw_signal(video_path, roi)
    plot_ppw_signal(ppw_signal)

if __name__ == "__main__":
    main()
