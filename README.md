# Real-Time-Vehicle-Detection-and-Counting

This project leverages YOLOv8 object detection and OpenCV to detect and count cars in a video. It utilizes a masked detection area to focus on specific regions of interest and tracks car movements to count vehicles crossing a defined line.

---

## Features

1. **Real-Time Detection**:
   - Detects cars in a video feed using YOLOv8 pre-trained weights.
2. **Masked Detection Area**:
   - Restricts detection to a specified region using a custom mask.
3. **Vehicle Counting**:
   - Counts cars as they cross a predefined counting line in the video.
4. **Visual Enhancements**:
   - Displays bounding boxes, class labels, and confidence scores for detected objects.

---

## Requirements

The required libraries and dependencies are listed in the `requirements.txt` file. Ensure you have Python 3.8+ installed. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup

**Step 1**: Clone the Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/rishraks/Real-Time-Vehicle-Detection-and-Counting.git
cd CarCountingProject
```

**Step 2**: Download YOLOv8 Weights
Download the pre-trained YOLOv8 weights (yolov8l.pt) from the Ultralytics GitHub Repository.
Place the file in the YOLO_Weights directory.

**Step 3**: Add Video and Mask

- Video:
  Place the video file (e.g., cars.mp4) in the Videos directory.
- Mask:
  Create a custom mask image (mask.png) that defines the detection region and place it in the Images directory. The mask should match the video resolution.

## Usage

Run the script with the following command:

```bash
python Counting_Car.py
```

## How It Works

1. Video Processing
   - The script reads the video frame-by-frame using OpenCV.
2. Mask Application
   - A mask is applied to restrict detection to a specific region.
3. Object Detection
   - YOLOv8 detects objects in the masked region. Only objects classified as "car" are processed further.
4. Vehicle Tracking and Counting
   - The script tracks the positions of detected cars.
   - A horizontal counting line is defined in the frame.
   - Cars crossing the line in the downward direction are counted.

## Limitations

- Occlusion: May face challenges in detecting overlapping cars.
- Lighting Conditions: Performance may vary based on video quality and lighting.
- Custom Regions: The accuracy of detection depends on proper masking.

## License

This project is licensed under the GPL General Public License v3.0. You are free to use, modify, and share this project under the terms of this license. See the LICENSE file for details.

## Acknowledgments
- YOLOv8: Object detection by Ultralytics.
- OpenCV: For efficient video processing.
- cvzone: For visualization and utility functions.
