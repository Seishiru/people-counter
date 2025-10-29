---

# 👥 People Counting Using YOLOv8 and OpenCV

This project uses **YOLOv8** object detection and **OpenCV tracking** to count the number of people entering or exiting through a doorway (or across a defined line) in a video feed.
It can be used for **crowd analytics, shop entrances, or security monitoring**.

---

## 🧠 Overview

The system detects people in each video frame using **Ultralytics YOLOv8**, assigns them unique IDs via a custom **object tracker**, and counts how many people cross a specific line in either direction (Up / Down).

### ✨ Features

* 🚶 Detects and tracks multiple people simultaneously
* 🔄 Counts people **entering and exiting** across a line
* 🧩 Works with **pre-recorded videos** or **live camera feed**
* 💾 Exports an annotated video with tracking and count overlay
* ⚡ Built with **Python**, **OpenCV**, **cvzone**, and **YOLOv8**

---

## 📁 Project Structure

```
PeopleCounting-ComputerVision/
│
├── 📁 data/              → Input videos (e.g., 3.mp4)
├── 📁 models/            → Pretrained YOLOv8 weights (yolov8s.pt)
├── 📁 outputs/           → Processed videos with annotations
├── 📁 src/               → Source code
│   └── countingYolov8.py (Main script)
├── 📁 utils/             → Helper files (tracker, coco.names)
├── requirements.txt      → Python dependencies
└── README.md             → Project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/PeopleCounting-ComputerVision.git
cd PeopleCounting-ComputerVision
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # For Windows
# source venv/bin/activate   # For macOS/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Example of what’s inside `requirements.txt`:

```
ultralytics
opencv-python
cvzone
pandas
numpy
```

---

## 🚀 Running the Project

### ▶️ From a video file

Place your input video (e.g., `3.mp4`) in the `data/` folder.

Then run:

```bash
cd src
python countingYolov8.py
```

* Input video: `data/3.mp4`
* Output video: `outputs/output_final.avi`

You’ll see:

* A live display of the video feed
* People being tracked with bounding boxes
* “Up” and “Down” counts shown on screen

---

## 🎥 Switching to Live Camera Mode (Optional)

To use your webcam instead of a video file, edit this line in `countingYolov8.py`:

```python
# cap = cv2.VideoCapture(os.path.join(BASE_DIR, 'data', '3.mp4'))
cap = cv2.VideoCapture(0)  # Use webcam instead
```

Then re-run the script:

```bash
python countingYolov8.py
```

---

## 📊 Output Example

* Bounding boxes drawn on each person
* Tracking IDs assigned dynamically
* Counter display at top-left corner
* Saved annotated video at `outputs/output_final.avi`

---

## 👩‍💻 Team Notes

* You can adjust the **counting lines** (`cy1`, `cy2`) in the script to fit your camera angle.
* The **model weight** (`yolov8s.pt`) is small and fast — you can replace it with `yolov8m.pt` or `yolov8l.pt` for better accuracy.
* The tracker logic is in `utils/tracker.py`.

---

## 🧾 License

This project is open-source under the **MIT License**.
Feel free to modify and use it for academic or commercial purposes.

---
