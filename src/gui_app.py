import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
from torchgeo.models import resnet50
import os

class SolarPanelDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("🌞 Solar Panel Detector")
        self.root.geometry("800x700")
        self.root.resizable(True, True)

        # Center the window
        self.center_window()

        # Model setup
        self.device = self.setup_device()
        self.model = None
        self.transform = self.setup_transform()
        self.class_names = ['no_solar', 'solar']
        self.current_image_path = None

        # Create UI
        self.create_widgets()

        # Load model if available
        self.load_model()

    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def setup_device(self):
        """Setup device for inference"""
        if torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon) for inference")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("Using CUDA for inference")
            return torch.device("cuda")
        else:
            print("Using CPU for inference")
            return torch.device("cpu")

    def setup_transform(self):
        """Setup image transforms for the model"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_model(self):
        """Load the trained model"""
        model_path = 'solar_panel_model_m1.pth'

        if not os.path.exists(model_path):
            print(f"❌ Model file '{model_path}' not found!")
            self.show_status("No model found. Please train the model first!", "error")
            return False

        try:
            # Initialize model
            self.model = resnet50(pretrained=False)  # Don't load pretrained weights
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 2 classes

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            accuracy = checkpoint.get('accuracy', 0)
            self.show_status(f"✅ Model loaded! Test Accuracy: {accuracy:.2f}%", "success")
            print(f"Model loaded successfully with {accuracy:.2f}% test accuracy")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.show_status(f"Error loading model: {str(e)}", "error")
            return False

    def create_widgets(self):
        """Create the main UI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="🌞 Solar Panel Detector",
                                font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Image display frame
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        self.image_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)

        # Image label (for displaying the image)
        self.image_label = ttk.Label(self.image_frame, text="No image loaded",
                                     background="white", relief="sunken",
                                     font=('Arial', 12))
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)

        # Load image button
        self.load_btn = ttk.Button(button_frame, text="📁 Load Image",
                                   command=self.load_image)
        self.load_btn.grid(row=0, column=0, padx=(0, 10))

        # Clear button
        clear_btn = ttk.Button(button_frame, text="🗑️ Clear",
                               command=self.clear_image)
        clear_btn.grid(row=0, column=1, padx=(0, 10))

        # Predict button
        self.predict_btn = ttk.Button(button_frame, text="🔮 Predict",
                                      command=self.predict_image, state="disabled")
        self.predict_btn.grid(row=0, column=2)

        # Instructions
        instructions = ttk.Label(main_frame,
                                 text="Load an image using the button above, then click Predict to analyze for solar panels!",
                                 font=('Arial', 10), foreground="gray")
        instructions.grid(row=3, column=0, columnspan=3, pady=10)

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="15")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        results_frame.columnconfigure(0, weight=1)

        # Result label
        self.result_label = ttk.Label(results_frame,
                                      text="Load an image to get predictions",
                                      font=('Arial', 12), foreground="gray")
        self.result_label.grid(row=0, column=0, sticky=tk.W)

        # Confidence progress bar
        self.confidence_var = tk.DoubleVar()
        self.confidence_progress = ttk.Progressbar(results_frame,
                                                   variable=self.confidence_var,
                                                   maximum=100, length=300)
        self.confidence_progress.grid(row=1, column=0, sticky=tk.W, pady=(10, 0))

        # Confidence label
        self.confidence_label = ttk.Label(results_frame, text="",
                                          font=('Arial', 10, 'bold'))
        self.confidence_label.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))

        # Detailed probabilities frame
        probs_frame = ttk.Frame(results_frame)
        probs_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        probs_frame.columnconfigure(0, weight=1)
        probs_frame.columnconfigure(1, weight=1)

        # Solar probability
        ttk.Label(probs_frame, text="☀️ Solar Panels:",
                  font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.solar_prob_label = ttk.Label(probs_frame, text="0%",
                                          font=('Arial', 9), foreground="green")
        self.solar_prob_label.grid(row=0, column=1, sticky=tk.E)

        # No solar probability
        ttk.Label(probs_frame, text="🏠 No Solar:",
                  font=('Arial', 9, 'bold')).grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.nosolar_prob_label = ttk.Label(probs_frame, text="0%",
                                            font=('Arial', 9), foreground="red")
        self.nosolar_prob_label.grid(row=1, column=1, sticky=tk.E, pady=(5, 0))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                               relief="sunken", anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

    def load_image(self):
        """Load image via file dialog"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select an image for solar panel detection",
            filetypes=filetypes
        )

        if filename:
            self.process_image(filename)

    def is_image_file(self, filename):
        """Check if file is a valid image"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return os.path.splitext(filename)[1].lower() in valid_extensions

    def process_image(self, filename):
        """Process the selected image"""
        try:
            # Load and display image
            self.current_image_path = filename
            image = Image.open(filename)

            # Store original for prediction
            self.original_image = image.copy()

            # Resize for display (max 400x400)
            display_image = image.copy()
            display_image.thumbnail((400, 400), Image.Resampling.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_image)

            # Update image display
            filename_short = os.path.basename(filename)
            self.image_label.configure(
                image=photo,
                text=f"{filename_short}",
                compound=tk.TOP,
                font=('Arial', 10)
            )
            self.image_label.image = photo  # Keep a reference

            # Enable predict button
            self.predict_btn.configure(state="normal")

            # Update status
            self.show_status(f"Image loaded: {filename_short}", "info")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{str(e)}")
            self.show_status(f"Error loading image: {str(e)}", "error")

    def clear_image(self):
        """Clear the current image"""
        self.image_label.configure(image="", text="No image loaded", compound=tk.CENTER)
        self.predict_btn.configure(state="disabled")
        self.result_label.configure(text="Load an image to get predictions", foreground="gray")
        self.confidence_var.set(0)
        self.confidence_progress['value'] = 0
        self.confidence_label.configure(text="")
        self.solar_prob_label.configure(text="0%")
        self.nosolar_prob_label.configure(text="0%")
        self.show_status("Image cleared", "info")
        self.current_image_path = None
        self.original_image = None

    def predict_image(self):
        """Run prediction on the loaded image"""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded! Please train the model first.")
            return

        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        try:
            self.show_status("🔄 Running prediction...", "info")
            self.root.update()

            # Load and preprocess image
            image = self.original_image.convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                predicted_class = self.class_names[predicted.item()]

            # Update UI with results
            confidence_pct = confidence.item() * 100
            solar_conf = probabilities[1].item() * 100
            nosolar_conf = probabilities[0].item() * 100

            # Update confidence progress bar
            self.confidence_var.set(confidence_pct)
            self.confidence_progress['value'] = confidence_pct

            # Color coding for results
            if predicted_class == 'solar':
                result_color = "#2E7D32"  # Dark green
                emoji = "☀️"
                result_text = f"{emoji} SOLAR PANELS DETECTED!"
                confidence_color = "#388E3C" if confidence_pct > 80 else "#FBC02D"
            else:
                result_color = "#C62828"  # Dark red
                emoji = "🏠"
                result_text = f"{emoji} NO SOLAR PANELS DETECTED"
                confidence_color = "#D32F2F" if confidence_pct > 80 else "#FBC02D"

            # Update main result
            self.result_label.configure(
                text=result_text,
                foreground=result_color,
                font=('Arial', 14, 'bold')
            )

            # Update confidence label
            self.confidence_label.configure(
                text=f"Confidence: {confidence_pct:.1f}%",
                foreground=confidence_color
            )

            # Update detailed probabilities
            self.solar_prob_label.configure(text=f"{solar_conf:.1f}%")
            self.nosolar_prob_label.configure(text=f"{nosolar_conf:.1f}%")

            # Color code probability labels based on values
            self.solar_prob_label.configure(foreground="#4CAF50" if solar_conf > 50 else "gray")
            self.nosolar_prob_label.configure(foreground="#F44336" if nosolar_conf > 50 else "gray")

            self.show_status(f"✅ Prediction complete: {predicted_class} ({confidence_pct:.1f}% confidence)", "success")

            print(f"Prediction: {predicted_class} ({confidence_pct:.1f}% confidence)")
            print(f"  Solar: {solar_conf:.1f}%, No Solar: {nosolar_conf:.1f}%")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction:\n{str(e)}")
            self.show_status(f"❌ Prediction error: {str(e)}", "error")

    def show_status(self, message, msg_type="info"):
        """Update status bar with message"""
        # Color coding for status (via text color)
        status_colors = {
            "success": "green",
            "error": "red",
            "info": "blue",
            "warning": "orange"
        }

        color = status_colors.get(msg_type, "black")
        self.status_var.set(f"{message}")

        # Brief flash effect by changing color temporarily
        def reset_color():
            self.status_var.set(message)

        self.root.after(2000, reset_color)

    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the Solar Panel Detector?"):
            self.root.destroy()

def main():
    """Main function"""
    root = tk.Tk()
    app = SolarPanelDetector(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the application
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        root.destroy()

if __name__ == '__main__':
    main()