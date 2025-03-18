import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)  # Open the image
        img = img.resize((300, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        label_img.config(image=img_tk)
        label_img.image = img_tk  # Keep a reference to avoid garbage collection

# Create the main window
root = tk.Tk()
root.title("UNO Card Detection")
root.geometry("400x400")

# Add an "Upload Image" button
btn_upload = tk.Button(root, text="Upload Image", command=upload_image)
btn_upload.pack(pady=20)

# Label to display the uploaded image
label_img = tk.Label(root)
label_img.pack()

# Run the application
root.mainloop()
