import tkinter as tk
from tkinter import messagebox
from ui import run_webcam_mode, run_image_mode
from feature_matching import load_templates
import cv2

def launch_webcam():
    root.destroy()  # Close the UI window
    run_webcam_mode(sift, bf, templates)

def launch_image():
    root.destroy()  # Close the UI window
    run_image_mode(sift, bf, templates)

# Load SIFT and templates once for both modes
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
templates = load_templates()

# Set up UI
root = tk.Tk()
root.title("UNO Card Detector 🎴")
root.geometry("300x200")
root.configure(bg="#2C2F33")

title = tk.Label(root, text="Select Mode", font=("Helvetica", 16, "bold"), fg="white", bg="#2C2F33")
title.pack(pady=20)

webcam_btn = tk.Button(root, text="📷 Webcam", font=("Helvetica", 12), width=20, bg="#7289DA", fg="white", command=launch_webcam)
webcam_btn.pack(pady=5)

upload_btn = tk.Button(root, text="🖼️ Upload Image", font=("Helvetica", 12), width=20, bg="#43B581", fg="white", command=launch_image)
upload_btn.pack(pady=5)

root.mainloop()
