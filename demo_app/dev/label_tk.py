### LABELLING APP PROTOTYPE USING TK
### NOT LIKELY TO BE TOUCHED
####################################


import cv2
import os
import json
import supervision as sv
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import re


class ImageLabeller:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Labeller")
        self.model = YOLO("yolov8n.pt")
        self.image_files = []
        self.detections = {}
        self.current_index = 0
        self.boxes = []
        self.drawing = False
        self.start_x = None
        self.start_y = None

        # UI Elements
        self.frame = tk.Frame(root)
        self.frame.pack()

        self.canvas = tk.Canvas(self.frame, width=800, height=600)
        self.canvas.grid(row=0, column=0)

        self.class_listbox = tk.Listbox(self.frame, width=20, height=30)
        self.class_listbox.grid(row=0, column=1, padx=10, sticky="ns")
        self.class_listbox.bind("<Double-Button-1>", self.delete_selected_box)

        self.label_filename = tk.Label(root, text="No image loaded", font=("Arial", 12))
        self.label_filename.pack()

        self.btn_load = tk.Button(root, text="Load Images", command=self.load_images)
        self.btn_load.pack()

        self.btn_prev = tk.Button(
            root, text="Previous", command=self.previous_image, state=tk.DISABLED
        )
        self.btn_prev.pack()

        self.btn_next = tk.Button(
            root, text="Next", command=self.next_image, state=tk.DISABLED
        )
        self.btn_next.pack()

        self.btn_export = tk.Button(
            root, text="Export Labels", command=self.export_labels
        )
        self.btn_export.pack()

        self.btn_close = tk.Button(root, text="Close", command=root.quit)
        self.btn_close.pack()

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_box)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

    def load_images(self):
        folder_selected = filedialog.askdirectory(title="Select Image Folder")
        if not folder_selected:
            return

        def numerical_sort(filename):
            return [
                int(text) if text.isdigit() else text
                for text in re.split(r"(\d+)", filename)
            ]

        self.image_files = sorted(
            [
                os.path.join(folder_selected, f)
                for f in os.listdir(folder_selected)
                if f.endswith((".jpg", ".png"))
            ],
            key=lambda x: numerical_sort(os.path.basename(x)),
        )
        if not self.image_files:
            messagebox.showerror("Error", "No images found in the selected folder.")
            return

        self.btn_next["state"] = tk.NORMAL
        self.btn_prev["state"] = tk.NORMAL
        self.current_index = 0
        self.load_image()

    def load_image(self):
        if self.current_index >= len(self.image_files):
            self.current_index = len(self.image_files)

        image_path = self.image_files[self.current_index]
        self.label_filename.config(text=f"Current file: {os.path.basename(image_path)}")
        self.current_image = Image.open(image_path)
        self.current_image.thumbnail((800, 600))
        self.tk_image = ImageTk.PhotoImage(self.current_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        if image_path not in self.detections:
            self.run_detection(image_path)
        else:
            self.boxes = self.detections[os.path.basename(image_path)]
            self.update_ui()

    def run_detection(self, image_path):
        results = self.model(image_path)
        self.boxes = []
        for result in results:
            for box, conf, cls in zip(
                result.boxes.xyxy, result.boxes.conf, result.boxes.cls
            ):
                bbox = box.tolist()
                self.boxes.append(
                    {"bbox": bbox, "confidence": float(conf), "class": int(cls)}
                )
        self.detections[os.path.basename(image_path)] = self.boxes
        self.update_ui()

    def update_ui(self):
        self.canvas.delete("all")
        self.class_listbox.delete(0, tk.END)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        for box in self.boxes:
            self.draw_existing_box(box["bbox"], box["class"])
            self.class_listbox.insert(tk.END, f"Class {box['class']}: {box['bbox']}")

    def draw_existing_box(self, bbox, cls):
        x1, y1, x2, y2 = bbox
        self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="red", width=2, tags=f"box_{cls}"
        )
        text_y = y1 - 10 if y1 > 20 else y1 + 15
        self.canvas.create_text(
            (x1 + x2) / 2,
            text_y,
            text=str(cls),
            fill="yellow",
            font=("Arial", 12, "bold"),
            tags=f"text_{cls}",
        )

    def start_draw(self, event):
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y

    def draw_box(self, event):
        if self.drawing:
            self.canvas.delete("temp_box")
            self.canvas.create_rectangle(
                self.start_x,
                self.start_y,
                event.x,
                event.y,
                outline="blue",
                width=2,
                tags="temp_box",
            )

    def end_draw(self, event):
        if self.drawing:
            self.drawing = False
            bbox = [self.start_x, self.start_y, event.x, event.y]
            new_class = simpledialog.askinteger(
                "New Class", "Enter class ID for new box:"
            )
            if new_class is not None:
                self.boxes.append({"bbox": bbox, "confidence": 1.0, "class": new_class})
                self.detections[
                    os.path.basename(self.image_files[self.current_index])
                ] = self.boxes
                self.update_ui()

    def delete_selected_box(self, event):
        selection = self.class_listbox.curselection()
        if selection:
            index = selection[0]
            del self.boxes[index]
            self.class_listbox.delete(index)
            self.detections[os.path.basename(self.image_files[self.current_index])] = (
                self.boxes
            )
            self.update_ui()

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def next_image(self):
        self.current_index += 1
        self.load_image()

    def export_labels(self):
        for image_path in self.image_files:
            image_name = os.path.basename(
                image_path
            )  # Keep the full filename with extension
            label_file = os.path.join(
                os.path.dirname(image_path), f"{os.path.splitext(image_name)[0]}.txt"
            )
            image = Image.open(image_path)
            width, height = image.size

            if (
                image_name in self.detections
            ):  # Ensure we're checking with the correct key
                with open(label_file, "w") as f:
                    for box in self.detections[
                        image_name
                    ]:  # Now correctly accesses detections
                        x1, y1, x2, y2 = box["bbox"]
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        f.write(
                            f"{box['class']} {x_center} {y_center} {box_width} {box_height}\n"
                        )

        messagebox.showinfo("Export Successful", "Labels exported in YOLO format!")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabeller(root)
    root.mainloop()
