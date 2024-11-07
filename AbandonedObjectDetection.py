import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import time

class AbandonedObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Abandoned Object Detection")
        self.root.geometry("800x700")

        self.video_paths = []
        self.current_video_index = 0
        self.cap = None

        self.stabilized = False
        self.background_model = None
        self.first_frame = None

        self.suspicious_duration = 30000  # 30 seconds in milliseconds
        self.video_start_time = None
        self.popup_shown = False

        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.load_video_button = tk.Button(root, text="Load Videos", command=self.load_videos)
        self.load_video_button.pack(pady=10)

        self.after_id = None

    def load_videos(self):
        self.video_paths = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4;*.avi")])
        if self.video_paths:
            self.start_button["state"] = tk.NORMAL

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(img)
        self.video_label.config(image=img)
        self.video_label.image = img

    def start_detection(self):
        self.start_button["state"] = tk.DISABLED
        self.stop_button["state"] = tk.NORMAL

        if self.video_paths:
            self.load_video()

    def load_video(self):
        if self.current_video_index < len(self.video_paths):
            video_path = self.video_paths[self.current_video_index]
            self.cap = cv2.VideoCapture(video_path)

            if not self.cap.isOpened():
                print(f"Error: Unable to open video file: {video_path}")
                self.cap = None
                return

            ret, first_frame = self.cap.read()
            if ret:
                self.first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                self.stabilized = False
                self.video_start_time = time.time() * 1000  # Convert to milliseconds
                self.popup_shown = False
            else:
                print("Error: Unable to read the first frame.")
                self.cap.release()
                self.cap = None
                self.start_button["state"] = tk.DISABLED

            self.process_frame()
        else:
            self.stop_detection()

    def process_frame(self):
        ret, frame = self.cap.read()

        if ret:
            if not self.stabilized:
                if self.first_frame is None:
                    return

                diff = cv2.absdiff(self.first_frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                _, thresholded = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

                if cv2.countNonZero(thresholded) / thresholded.size < 0.01:
                    self.background_model = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.stabilized = True
            else:
                binary_mask = self.perform_background_subtraction(frame)
                contours = self.find_contours(binary_mask)
                filtered_contours = self.filter_contours(contours)
                frame_with_contours = frame.copy()
                cv2.drawContours(frame_with_contours, filtered_contours, -1, (0, 255, 0), 2)
                self.display_frame(frame_with_contours)

                current_time = time.time() * 1000  # Convert to milliseconds
                elapsed_time = current_time - self.video_start_time

                if not self.popup_shown and elapsed_time >= self.suspicious_duration:
                    messagebox.showwarning("Suspicious Object Detected", f"Suspicious object detected in video {self.current_video_index + 1}!")
                    self.popup_shown = True

            self.after_id = self.root.after(30, self.process_frame)
        else:
            self.stop_detection()

    def stop_detection(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        if self.cap is not None:
            self.cap.release()
            self.video_label.config(image=None)

        self.first_frame = None
        self.background_model = None
        self.stabilized = False

        self.start_button["state"] = tk.NORMAL
        self.stop_button["state"] = tk.DISABLED

        self.current_video_index += 1
        self.load_video()

    def perform_background_subtraction(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(self.background_model, gray)
        _, thresholded = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        return thresholded

    def find_contours(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def filter_contours(self, contours):
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]
        return filtered_contours

if __name__ == "__main__":
    root = tk.Tk()
    app = AbandonedObjectDetectionApp(root)
    root.mainloop()
