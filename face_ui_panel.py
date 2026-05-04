import cv2
import threading
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk
from ultralytics import YOLO


MODEL_PATH = "runs/classify/train4/weights/best.pt" #use best training for identifying
DEFAULT_IMAGE_PATH = "test{i}.jpg"
OUTPUT_IMAGE_PATH = "processed_output.jpg"
CROP_PATH = "cropped_face.jpg"


class FaceRecognitionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition - Detection + Classification")
        self.root.geometry("1100x650")
        self.root.configure(bg="#101010")

        self.current_image_path = DEFAULT_IMAGE_PATH
        self.displayed_image = None

        self.build_ui()

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="Face Recognition Detection + Classification",
            bg="#101010",
            fg="white",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=10)

        main_frame = tk.Frame(self.root, bg="#101010")
        main_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # LEFT PANEL: Image preview
        image_panel = tk.Frame(main_frame, bg="#1b1b1b", bd=2, relief="ridge")
        image_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        image_title = tk.Label(
            image_panel,
            text="Processed Image",
            bg="#1b1b1b",
            fg="#00ff66",
            font=("Arial", 16, "bold")
        )
        image_title.pack(pady=10)

        self.image_label = tk.Label(
            image_panel,
            bg="#1b1b1b",
            text="Run the model to show image here",
            fg="gray",
            font=("Arial", 14)
        )
        self.image_label.pack(fill="both", expand=True, padx=10, pady=10)

        # RIGHT PANEL: Terminal output
        terminal_panel = tk.Frame(main_frame, bg="#000000", bd=2, relief="ridge")
        terminal_panel.pack(side="right", fill="both", expand=True, padx=(10, 0))

        terminal_title = tk.Label(
            terminal_panel,
            text="Terminal Output",
            bg="#000000",
            fg="#00ff66",
            font=("Arial", 16, "bold")
        )
        terminal_title.pack(pady=10)

        self.terminal = tk.Text(
            terminal_panel,
            bg="#000000",
            fg="#00ff66",
            insertbackground="#00ff66",
            font=("Courier New", 12),
            wrap="word"
        )
        self.terminal.pack(fill="both", expand=True, padx=10, pady=10)

        # BUTTONS
        button_frame = tk.Frame(self.root, bg="#101010")
        button_frame.pack(pady=10)

        choose_btn = tk.Button(
            button_frame,
            text="Choose Image",
            command=self.choose_image,
            font=("Arial", 12, "bold"),
            bg="#333333",
            fg="white",
            padx=20,
            pady=8
        )
        choose_btn.pack(side="left", padx=10)

        run_btn = tk.Button(
            button_frame,
            text="Run Detection + Classification",
            command=self.start_processing_thread,
            font=("Arial", 12, "bold"),
            bg="#00aa44",
            fg="white",
            padx=20,
            pady=8
        )
        run_btn.pack(side="left", padx=10)

        clear_btn = tk.Button(
            button_frame,
            text="Clear Terminal",
            command=self.clear_terminal,
            font=("Arial", 12, "bold"),
            bg="#444444",
            fg="white",
            padx=20,
            pady=8
        )
        clear_btn.pack(side="left", padx=10)

    def log(self, message):
        self.terminal.insert("end", message + "\n")
        self.terminal.see("end")
        self.root.update_idletasks()

    def clear_terminal(self):
        self.terminal.delete("1.0", "end")

    def choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_image_path = file_path


    def start_processing_thread(self):
        thread = threading.Thread(target=self.process_image)
        thread.daemon = True
        thread.start()

    def process_image(self):
        try:
            self.log("[INFO] Starting face recognition pipeline...")

            if not Path(MODEL_PATH).exists():
                self.log("[ERROR] Model file not found.")
                self.log("[FIX] Check that your best.pt path is correct.")
                return

            model = YOLO(MODEL_PATH)

            self.log("[INFO] Loading OpenCV face detector...")
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            self.log(f"[INFO] Loading image: {self.current_image_path}")
            img = cv2.imread(self.current_image_path)

            if img is None:
                self.log("[ERROR] Could not load image.")
                return

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            self.log("[INFO] Detecting faces...")
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(30, 30)
            )

            self.log(f"[INFO] Faces detected: {len(faces)}")

            if len(faces) == 0:
                self.log("[WARNING] No face detected.")
                self.log("[INFO] Using the full image instead.")
                
                crop_path = CROP_PATH
                cv2.imwrite(crop_path, img)

            else:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]

                face_crop= img[y:y+h, x:x+w]

                crop_path = "cropped_face.jpg"
                cv2.imwrite(crop_path, face_crop)

            # Use largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            self.log(f"[INFO] Largest face box: x={x}, y={y}, w={w}, h={h}")

            face_crop = img[y:y + h, x:x + w]
            cv2.imwrite(CROP_PATH, face_crop)
            self.log(f"[INFO] Cropped face saved as: {CROP_PATH}")

            self.log("[INFO] Classifying cropped face with YOLO...")
            results = model(CROP_PATH)

            predicted_class_id = results[0].probs.top1
            confidence = results[0].probs.top1conf.item()

            if confidence < 0.50:
                predicted_identity = "Unknown"
            else:
                predicted_identity = results[0].names[predicted_class_id]

            self.log(f"[RESULT] Predicted identity/class: {predicted_identity}")
            self.log(f"[RESULT] Confidence: {round(confidence * 100, 2)}%")

            # Draw result on image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            label = f"{predicted_identity} ({confidence * 100:.1f}%)"
            cv2.putText(
                img,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imwrite(OUTPUT_IMAGE_PATH, img)
            self.log(f"[INFO] Processed image saved as: {OUTPUT_IMAGE_PATH}")

            self.show_processed_image(OUTPUT_IMAGE_PATH)

        except Exception as e:
            self.log(f"[ERROR] {e}")

    def show_processed_image(self, image_path):
        image = Image.open(image_path)

        # Resize image to fit panel
        max_width = 520
        max_height = 480
        image.thumbnail((max_width, max_height))

        self.displayed_image = ImageTk.PhotoImage(image)

        self.image_label.config(
            image=self.displayed_image,
            text=""
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionUI(root)
    root.mainloop()