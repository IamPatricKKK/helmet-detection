"""
Data collection application for helmet detection
Uses camera to capture images and manual classification
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time
from datetime import datetime


class DataCollectionApp:
    """Data collection application for helmet detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Helmet Detection Data Collection")
        self.root.geometry("1000x700")
        
        # Initialize camera
        self.cap = None
        self.camera_running = False
        
        # Create data directories
        self.data_folder = "data_collection"
        self.helmet_folder = os.path.join(self.data_folder, "with_helmet")
        self.no_helmet_folder = os.path.join(self.data_folder, "no_helmet")
        
        # Create directories if they don't exist
        for folder in [self.helmet_folder, self.no_helmet_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        # Count variables
        self.helmet_count = len(os.listdir(self.helmet_folder)) if os.path.exists(self.helmet_folder) else 0
        self.no_helmet_count = len(os.listdir(self.no_helmet_folder)) if os.path.exists(self.no_helmet_folder) else 0
        
        # Create UI
        self.create_widgets()
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def create_widgets(self):
        """Create UI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Helmet Detection Data Collection", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Camera control frame
        control_frame = ttk.LabelFrame(main_frame, text="Camera Control", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_btn = ttk.Button(control_frame, text="Start Camera", 
                                   command=self.start_camera)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Camera", 
                                  command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        image_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="No data", 
                                    background="white", anchor="center")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Data collection frame
        data_frame = ttk.LabelFrame(main_frame, text="Data Collection", padding="10")
        data_frame.grid(row=2, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                       padx=(10, 0), pady=(0, 10))
        
        # Statistics
        ttk.Label(data_frame, text="Statistics:", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(data_frame, text="With helmet:").grid(row=1, column=0, sticky=tk.W)
        self.helmet_count_label = ttk.Label(data_frame, text=str(self.helmet_count), 
                                           foreground="green", font=("Arial", 12, "bold"))
        self.helmet_count_label.grid(row=1, column=1, padx=(10, 0))
        
        ttk.Label(data_frame, text="No helmet:").grid(row=2, column=0, sticky=tk.W)
        self.no_helmet_count_label = ttk.Label(data_frame, text=str(self.no_helmet_count), 
                                             foreground="red", font=("Arial", 12, "bold"))
        self.no_helmet_count_label.grid(row=2, column=1, padx=(10, 0))
        
        # Capture buttons
        ttk.Label(data_frame, text="Capture:", font=("Arial", 12, "bold")).grid(row=3, column=0, columnspan=2, pady=(20, 10))
        
        self.capture_helmet_btn = ttk.Button(data_frame, text="Capture - With Helmet", 
                                           command=self.capture_with_helmet, state="disabled")
        self.capture_helmet_btn.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.capture_no_helmet_btn = ttk.Button(data_frame, text="Capture - No Helmet", 
                                               command=self.capture_without_helmet, state="disabled")
        self.capture_no_helmet_btn.grid(row=5, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Information", padding="10")
        info_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=8, width=80, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        # Log startup information
        self.log_info("Data collection application started")
        self.log_info(f"With helmet folder: {self.helmet_folder}")
        self.log_info(f"No helmet folder: {self.no_helmet_folder}")
        self.log_info("Instructions: Start camera → Position person in frame → Capture image by classification")
    
    def log_info(self, message):
        """Log information"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.info_text.see(tk.END)
        self.root.update()
    
    def start_camera(self):
        """Start camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera!")
            return
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.camera_running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.capture_helmet_btn.config(state="normal")
        self.capture_no_helmet_btn.config(state="normal")
        
        self.log_info("Camera started")
        
        # Run camera loop
        thread = threading.Thread(target=self.camera_loop)
        thread.daemon = True
        thread.start()
    
    def camera_loop(self):
        """Camera loop"""
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            
            # Draw bounding boxes for faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            self.display_frame(frame)
    
    def display_frame(self, frame):
        """Display frame"""
        # Resize frame
        height, width = frame.shape[:2]
        max_width = 600
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update UI
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def capture_with_helmet(self):
        """Capture image with helmet"""
        if not self.camera_running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"helmet_{timestamp}.jpg"
        filepath = os.path.join(self.helmet_folder, filename)
        
        # Save image
        success = cv2.imwrite(filepath, frame)
        if success:
            self.helmet_count += 1
            self.helmet_count_label.config(text=str(self.helmet_count))
            self.log_info(f"Captured image with helmet: {filename}")
        else:
            self.log_info("Failed to save image")
    
    def capture_without_helmet(self):
        """Capture image without helmet"""
        if not self.camera_running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"no_helmet_{timestamp}.jpg"
        filepath = os.path.join(self.no_helmet_folder, filename)
        
        # Save image
        success = cv2.imwrite(filepath, frame)
        if success:
            self.no_helmet_count += 1
            self.no_helmet_count_label.config(text=str(self.no_helmet_count))
            self.log_info(f"Captured image without helmet: {filename}")
        else:
            self.log_info("Failed to save image")
    
    def stop_camera(self):
        """Stop camera"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.capture_helmet_btn.config(state="disabled")
        self.capture_no_helmet_btn.config(state="disabled")
        
        self.log_info("Camera stopped")


def main():
    """Main function"""
    root = tk.Tk()
    app = DataCollectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

