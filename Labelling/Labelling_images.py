import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import csv
import matplotlib.pyplot as plt


class ImageLabeler(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image Labeler")
        self.geometry("1000x600")

        self.labels = ["StartPlatL", "StepL", "StartPlatR", "StepR", "Door", "TransitionL", "TransitionR"]
        self.label_colors = self.generate_label_colors(self.labels)
        self.current_label = tk.StringVar(value=self.labels[0])

        self.points = {}

        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.scroll_x = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.scroll_y = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        self.image = None
        self.image_tk = None

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Shift-Button-3>", self.on_shift_right_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)

        self.dragging_point = None

        self.create_widgets()

    def create_widgets(self):
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, pady=5)

        open_button = tk.Button(button_frame, text="Open Image", command=self.open_image)
        open_button.pack(side=tk.LEFT, padx=5)

        save_button = tk.Button(button_frame, text="Save Points", command=self.save_points)
        save_button.pack(side=tk.LEFT, padx=5)

        delete_all_button = tk.Button(button_frame, text="Delete All Points", command=self.delete_all_points)
        delete_all_button.pack(side=tk.LEFT, padx=5)

        close_button = tk.Button(button_frame, text="Close", command=self.quit)
        close_button.pack(side=tk.LEFT, padx=5)

        instructions_label = tk.Label(control_frame, text="Instructions:\n"
                                                          "1. Right-click to place a point.\n"
                                                          "2. Shift + Right-click to delete the nearest point.\n"
                                                          "3. Left-click and drag to move a point.\n"
                                                          "4. Use the buttons to open/save images and delete points.")
        instructions_label.pack(side=tk.LEFT, padx=20)

        label_frame = tk.Frame(control_frame)
        label_frame.pack(side=tk.LEFT, padx=5)

        for label in self.labels:
            color = self.label_colors[label]
            label_subframe = tk.Frame(label_frame)
            label_subframe.pack(fill=tk.X, pady=5)

            color_box = tk.Label(label_subframe, bg=color, width=2)
            color_box.pack(side=tk.LEFT, padx=5)

            label_button = tk.Radiobutton(label_subframe, text=label, variable=self.current_label, value=label,
                                          indicatoron=0, width=20, command=self.update_current_label)
            label_button.pack(side=tk.LEFT)

    def generate_label_colors(self, labels):
        colormap = plt.get_cmap('hsv')
        colors = [colormap(i / len(labels)) for i in range(len(labels))]
        return {label: self.rgb_to_hex(color) for label, color in zip(labels, colors)}

    def rgb_to_hex(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def on_left_click(self, event):
        # Adjust for canvas scrolling
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        self.dragging_point = self.find_closest_point(x, y)
        if self.dragging_point:
            self.dragging_label = [label for label, point in self.points.items() if point[2] == self.dragging_point][0]

    def on_left_drag(self, event):
        if self.dragging_point:
            # Adjust for canvas scrolling
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.dragging_point, x - 3, y - 3, x + 3, y + 3)
            self.points[self.dragging_label] = (x, y, self.dragging_point)

    def on_right_click(self, event):
        # Adjust for canvas scrolling
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        label = self.current_label.get()
        color = self.label_colors[label]
        point = self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline=color)
        self.points[label] = (x, y, point)
        self.advance_label()

    def on_shift_right_click(self, event):
        # Adjust for canvas scrolling
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        closest_point = self.find_closest_point(x, y)
        if closest_point:
            self.canvas.delete(closest_point)
            label_to_delete = [label for label, point in self.points.items() if point[2] == closest_point]
            if label_to_delete:
                del self.points[label_to_delete[0]]

    def find_closest_point(self, x, y, max_distance=5):
        closest_point = None
        min_dist = float('inf')
        for px, py, point in [(*coords[:2], coords[2]) for coords in self.points.values()]:
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist < min_dist and dist <= max_distance:
                closest_point = point
                min_dist = dist
        return closest_point

    def advance_label(self):
        current_index = self.labels.index(self.current_label.get())
        next_index = (current_index + 1) % len(self.labels)
        if next_index != 0 or len(self.points) < len(self.labels):
            self.current_label.set(self.labels[next_index])
        else:
            self.current_label.set('')  # No more labels to advance to

    def update_current_label(self):
        print(f"Current label is now {self.current_label.get()}")

    def save_points(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["x", "y", "label"])
                for label, (x, y, point) in self.points.items():
                    writer.writerow([round(x, 2), round(y, 2), label])

    def delete_all_points(self):
        for label, (x, y, point) in self.points.items():
            self.canvas.delete(point)
        self.points.clear()


if __name__ == "__main__":
    app = ImageLabeler()
    app.mainloop()
