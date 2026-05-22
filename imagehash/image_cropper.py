import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os


class ImageCropper:
    """Interactive image cropping tool with click-to-crop functionality."""

    def __init__(self, image_path=None, crop_size=500):
        """Initialize the image cropper.

        Args:
            image_path (str | None): Full path to the image file. If None, a file dialog opens.
            crop_size (int): Width and height of the crop region in pixels.
        """
        self.image_path = None
        self.crop_size = crop_size
        self.original_image = None
        self.photo_image = None
        self.crop_count = 0

        # Create main window
        self.root = tk.Tk()
        self.root.title("Image Cropper")
        self.root.geometry("900x700")

        # Menu bar
        menu_bar = tk.Menu(self.root)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image...", command=self.open_image_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu_bar)

        # Instructions label
        self.instructions_label = tk.Label(
            self.root,
            text=f"Click on the image to crop a {crop_size}×{crop_size}px region (top-left from click)",
            bg="lightgray",
            fg="black",
            font=("Arial", 10),
        )
        self.instructions_label.pack(fill=tk.X, padx=10, pady=5)

        # Create scrollable frame with canvas and scrollbars
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Horizontal scrollbar
        h_scrollbar = tk.Scrollbar(main_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Vertical scrollbar
        v_scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Canvas
        self.canvas = tk.Canvas(
            main_frame,
            bg="gray20",
            cursor="crosshair",
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=v_scrollbar.set,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure scrollbars
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)

        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down

        self.canvas.create_text(
            10,
            10,
            anchor=tk.NW,
            text="Open File → File → Open Image...",
            fill="white",
            font=("Arial", 12, "bold"),
            tags=("placeholder",),
        )

        self.root.after(100, lambda: self.load_image(image_path))

    def select_image(self):
        """Open a file dialog to select an image file."""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
            ("All files", "*.*"),
        ]
        return filedialog.askopenfilename(title="Open Image", filetypes=filetypes)

    def open_image_file(self):
        """Open image selection dialog and load the chosen image."""
        image_path = self.select_image()
        if image_path:
            self.load_image(image_path)

    def load_image(self, image_path=None):
        """Load the image into the canvas."""
        if image_path is None or not os.path.exists(image_path):
            image_path = self.select_image()
            if not image_path:
                return

        self.image_path = image_path
        self.original_image = Image.open(self.image_path)
        self.img_width, self.img_height = self.original_image.size
        self.photo_image = ImageTk.PhotoImage(self.original_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=(0, 0, self.img_width, self.img_height))
        self.canvas.image = self.photo_image
        self.root.title(f"Image Cropper - {os.path.basename(self.image_path)}")
        self.instructions_label.config(
            text=f"Click on the image to crop a {self.crop_size}×{self.crop_size}px region (top-left from click)"
        )

    def on_click(self, event):
        """Handle mouse click on canvas."""
        # Get canvas scroll offset
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Round to nearest integer
        x = int(canvas_x)
        y = int(canvas_y)

        # Validate click is within image bounds
        if x < 0 or y < 0 or x >= self.img_width or y >= self.img_height:
            messagebox.showwarning("Out of Bounds", "Click must be within the image area.")
            return

        # Compute crop box (x, y as top-left corner)
        left = x
        top = y
        right = min(x + self.crop_size, self.img_width)
        bottom = min(y + self.crop_size, self.img_height)

        # Warn if crop extends beyond image
        if right - left < self.crop_size or bottom - top < self.crop_size:
            messagebox.showwarning(
                "Crop Out of Bounds",
                f"Crop region extends beyond image.\n"
                f"Cropping {right - left}×{bottom - top}px instead.",
            )

        # Perform crop
        cropped = self.original_image.crop((left, top, right, bottom))

        # Save cropped image
        self.crop_count += 1
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = os.path.join(
            os.path.dirname(self.image_path), f"{base_name}_crop_{self.crop_count}.jpg"
        )
        cropped.save(output_path)
        messagebox.showinfo("Success", f"Cropped image saved to:\n{output_path}")

    def on_mouse_wheel(self, event):
        """Handle mouse wheel scrolling."""
        if event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(3, "units")
        elif event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-3, "units")

    def run(self):
        """Launch the cropper interface."""
        self.root.mainloop()


if __name__ == "__main__":
    image_path = r"C:\swetha\mfg-product-fingerprint\parts\sample_22.jpg"
    cropper = ImageCropper(image_path, crop_size=500)
    cropper.run()
