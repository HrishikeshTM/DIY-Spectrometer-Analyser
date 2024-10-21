import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive plots
import matplotlib.pyplot as plt
import threading
import time

class VideoCaptureApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.roi_selected = False
        self.first_cropped = None
        self.first_dist = None
        self.frame_buffer = []
        self.num_frames = 50  # Default value

    def capture_frames(self):
        capture_thread = threading.Thread(target=self._capture_frames)
        capture_thread.start()
        capture_thread.join()

    def _capture_frames(self):
        for i in range(self.num_frames):
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.append(frame)
                print(f"Captured frame {i + 1}/{self.num_frames}")
            else:
                print("Error: Could not read frame during capture.")
                break
            time.sleep(0.01)

    def calculate_color_distribution(self, frame):
        b_dist, g_dist, r_dist, i_dist = [], [], [], []
        for i in range(frame.shape[1]):
            b_val = np.mean(frame[:, i][:, 0])
            g_val = np.mean(frame[:, i][:, 1])
            r_val = np.mean(frame[:, i][:, 2])
            i_val = (r_val + b_val + g_val) / 3

            b_dist.append(b_val)
            g_dist.append(g_val)
            r_dist.append(r_val)
            i_dist.append(i_val)

        return b_dist, g_dist, r_dist, i_dist

    def compare_graphs(self, first_dist, second_dist):
        differences = [np.abs(np.array(first) - np.array(second)) for first, second in zip(first_dist, second_dist)]
        return differences

    def plot_distributions(self, first_dist, second_dist, differences):
        plt.close('all')  # Close any existing plots
        fig, axs = plt.subplots(4, 1, figsize=(14, 24))  # Further increase figure size
        fig.suptitle("Color Distribution Comparison", fontsize=16, y=0.98)  # Move title up

        colors = ['blue', 'green', 'red']
        labels = ['Blue', 'Green', 'Red']

        # Set common y-axis limits
        y_limits = (0, 255)  # Assuming intensity values range from 0 to 255

        for i in range(3):
            axs[i].plot(first_dist[i], color=colors[i], label='First ' + labels[i], linewidth=2)
            axs[i].plot(second_dist[i], color=colors[i], label='Second ' + labels[i], linestyle='--', linewidth=2)
            axs[i].set_title(f'{labels[i]} Channel Comparison', fontsize=12)
            axs[i].set_xlabel('Pixel Position', fontsize=10)
            axs[i].set_ylabel('Intensity', fontsize=10)
            axs[i].set_ylim(y_limits)  # Set the same y-axis limits
            axs[i].legend(loc="upper right", fontsize=8)
            axs[i].grid(True, linestyle='--', alpha=0.7)
            axs[i].tick_params(axis='both', which='major', labelsize=8)

        if differences:
            axs[3].plot(differences[0], color='navy', label='Differences (Blue Channel)', linewidth=2)
            axs[3].plot(differences[1], color='darkgreen', label='Differences (Green Channel)', linewidth=2)
            axs[3].plot(differences[2], color='darkred', label='Differences (Red Channel)', linewidth=2)
            axs[3].set_title('Differences in Color Channels', fontsize=12)
            axs[3].set_xlabel('Pixel Position', fontsize=10)
            axs[3].set_ylabel('Difference Intensity', fontsize=10)
            axs[3].set_ylim(0, max(differences[0].max(), differences[1].max(), differences[2].max()))  # Adjust y-limits for differences
            axs[3].legend(loc="upper right", fontsize=8)
            axs[3].grid(True, linestyle='--', alpha=0.7)
            axs[3].tick_params(axis='both', which='major', labelsize= 8)
        else:
            axs[3].set_title('No Differences to Display', fontsize=12)

        plt.tight_layout(pad=2.0)
        plt.show(block=True)

    def run(self):
        print("Welcome to the Spectrapure App!")
        print("Press 'r' to select a Region of Interest (ROI).")
        print("Press 'n' to set the number of frames to average.")
        print("Press 's' to capture and compare color distributions.")
        print("Press 'q' to quit the application.")
        print("Made by: Hrishikesh M. from the github repo by Kousheek Chakraborty")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            k = cv2.waitKey(1)

            if k & 0xFF == ord('r'):
                r = cv2.selectROI(frame)
                if r[2] == 0 or r[3] == 0:
                    print("ROI selection cancelled.")
                    self.roi_selected = False
                else:
                    self.roi_selected = True

            elif k & 0xFF == ord('n'):
                self.num_frames = int(input("Enter the number of frames to average: "))
                print(f"Number of frames set to {self.num_frames}")

            elif k & 0xFF == ord('s') and self.roi_selected:
                self.frame_buffer = []
                print(f"Capturing {self.num_frames} frames... Please wait.")
                self.capture_frames()
                if len(self.frame_buffer) == self.num_frames:
                    averaged_frame = np.mean(self.frame_buffer, axis=0).astype(np.uint8)
                    cropped = averaged_frame[int(r[1]): int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

                    b_dist, g_dist, r_dist, i_dist = self.calculate_color_distribution(cropped)

                    if self.first_cropped is None:
                        self.first_cropped = cropped
                        self.first_dist = (b_dist, g_dist, r_dist, i_dist)
                        print(f"First graph captured using {self.num_frames} frames. Press 's' again to capture the second graph.")
                    else:
                        second_dist = (b_dist, g_dist, r_dist, i_dist)
                        differences = self.compare_graphs(self.first_dist, second_dist)
                        self.plot_distributions(self.first_dist, second_dist, differences)

                        self.first_cropped = None
                        self.first_dist = None
                        print("Press 's' to capture the first graph again.")
                else:
                    print(f"Error: Could not capture {self.num_frames} frames.")

            elif k & 0xFF == ord('q'):
                break

            else:
                if self.roi_selected:
                    cv2.imshow('ROI', frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])
                cv2.imshow('Frame', frame)
                cv2.waitKey(30)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VideoCaptureApp()
    app.run()