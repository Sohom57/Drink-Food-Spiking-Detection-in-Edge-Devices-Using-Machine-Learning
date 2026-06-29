import time
import platform
import psutil
import cv2
import csv
import os
import subprocess
from datetime import datetime

class PerformanceEvaluator:
    def __init__(self, log_dir="logs", ai_model_names="YOLOv8n + InsightFace Buffalo_L + MediaPipe"):
        """Initializes the evaluator, prints specs, and sets up data logging."""
        self.log_dir = log_dir
        self.ai_model_names = ai_model_names
        
        # Create a folder for logs if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.print_system_metrics()
        self.setup_logging()

    def get_pc_model(self):
        """Attempts to fetch the physical hardware model of the PC."""
        try:
            if platform.system() == "Windows":
                # Uses Windows Management Instrumentation to get the PC model
                output = subprocess.check_output("wmic csproduct get name", shell=True)
                return output.decode().split('\n')[1].strip()
            elif platform.system() == "Darwin":
                # Uses sysctl on macOS
                output = subprocess.check_output("sysctl -n hw.model", shell=True)
                return output.decode().strip()
            else:
                return platform.machine()
        except Exception:
            return "Unknown / Custom Build"

    def print_system_metrics(self):
        """Fetches PC specifications, prints them, and saves to a text file."""
        pc_model = self.get_pc_model()
        
        specs = (
            f"{'='*55}\n"
            f" EDGE-AI HARDWARE & MODEL EVALUATION MATRIX\n"
            f"{'='*55}\n"
            f"AI Pipeline   : {self.ai_model_names}\n"
            f"{'-'*55}\n"
            f"Device Model  : {pc_model}\n"
            f"OS Platform   : {platform.system()} {platform.release()}\n"
            f"Processor     : {platform.processor()}\n"
            f"CPU Cores     : {psutil.cpu_count(logical=False)} Physical / {psutil.cpu_count(logical=True)} Logical\n"
            f"Total Memory  : {round(psutil.virtual_memory().total / (1024.**3), 2)} GB\n"
            f"{'='*55}\n"
        )
        print("\n" + specs)
        
        # Save static hardware and AI specs to a text file
        specs_path = os.path.join(self.log_dir, "hardware_specs.txt")
        with open(specs_path, "w") as f:
            f.write(specs)

    def setup_logging(self):
        """Initializes the CSV file for real-time performance tracking."""
        session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.log_dir, f"metrics_log_{session_time}.csv")
        
        # Create file and write the column headers (Now including the AI Pipeline)
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp", 
                "AI_Models", 
                "Processing_Time_Sec", 
                "FPS", 
                "CPU_Usage_Percent", 
                "RAM_Usage_Percent"
            ])

    def update_and_draw(self, img, loop_start_time):
        """Calculates, draws, and logs real-time edge-CPU performance metrics."""
        process_time = time.time() - loop_start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        
        # Gather live hardware metrics
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        
        # 1. Draw on the camera HUD
        cv2.putText(img, f"FPS: {int(fps)} | CPU: {cpu_usage}%", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # 2. Log data safely to the CSV file
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%H:%M:%S.%f")[:-3], # Current time with milliseconds
                self.ai_model_names,                         # The AI Models being evaluated
                round(process_time, 4),                      # Time taken to process the frame
                round(fps, 2),                               # Frames per second
                cpu_usage,                                   # CPU load
                ram_usage                                    # RAM load
            ])