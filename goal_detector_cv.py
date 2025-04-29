import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import timedelta
import argparse
import os

class GoalDetectionSystem:
    def __init__(self, video_path, output_dir='output', debug=False):
        """
        Initialize the goal detection system
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Directory to save outputs
            debug (bool): Whether to show debug visualizations
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.debug = debug
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Open the video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video loaded: {self.video_width}x{self.video_height} @ {self.fps} fps")
        print(f"Total frames: {self.frame_count}, Duration: {self.frame_count/self.fps:.2f} seconds")
        
        # Initialize variables for analysis
        self.frame_buffer = []
        self.buffer_size = int(self.fps * 2)  # 2 seconds buffer
        self.motion_scores = []
        self.detected_goals = []
        
        # Goal detection parameters
        self.celebration_threshold = 40.0  # Motion threshold for celebration detection
        self.quiet_threshold = 10.0  # Motion threshold for quiet periods
        self.celebration_duration = int(self.fps * 2)  # Number of frames celebration should last
        self.min_frames_between_goals = int(self.fps * 15)  # Minimum 15 seconds between goals
        self.last_goal_frame = -self.min_frames_between_goals  # Initialize with a negative value
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, 
            varThreshold=50, 
            detectShadows=False
        )

    def get_timestamp(self, frame_number):
        """Convert frame number to timestamp string"""
        seconds = frame_number / self.fps
        return str(timedelta(seconds=seconds))

    def analyze_frame(self, frame, frame_number):
        """
        Analyze a single frame for goal detection
        
        Args:
            frame: The current video frame
            frame_number: The current frame number
            
        Returns:
            tuple: (processed_frame, motion_score)
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Calculate motion score (percentage of foreground pixels)
        motion_score = (np.count_nonzero(fg_mask) / fg_mask.size) * 100
        
        # Create visualization for debugging
        if self.debug:
            debug_frame = frame.copy()
            # Add motion score text
            cv2.putText(
                debug_frame, 
                f"Motion: {motion_score:.2f}%", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Add frame number and timestamp
            cv2.putText(
                debug_frame, 
                f"Frame: {frame_number} | Time: {self.get_timestamp(frame_number)}", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0), 
                2
            )
            
            # Overlay foreground mask
            mask_overlay = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            debug_frame = cv2.addWeighted(debug_frame, 0.7, mask_overlay, 0.3, 0)
            
            return debug_frame, motion_score
        
        return frame, motion_score

    def detect_goal_events(self, current_frame, motion_scores, frame_number):
        """
        Detect goal events based on motion patterns
        
        Args:
            current_frame: Current video frame
            motion_scores: List of recent motion scores
            frame_number: Current frame number
            
        Returns:
            bool: True if a goal was detected
        """
        # Need enough frames in buffer to analyze
        if len(motion_scores) < self.celebration_duration:
            return False
        
        # Frames since last detected goal
        frames_since_last_goal = frame_number - self.last_goal_frame
        
        # Skip if we're too close to the previous goal
        if frames_since_last_goal < self.min_frames_between_goals:
            return False
        
        # Check for the celebration pattern:
        # 1. Current motion is high (celebration)
        current_motion = motion_scores[-1]
        
        # 2. Motion was relatively low before (quiet moment after shot)
        quiet_period = motion_scores[-self.celebration_duration:-self.celebration_duration//2]
        was_quiet = np.mean(quiet_period) < self.quiet_threshold
        
        # 3. Current motion is sustained (celebration continues)
        celebration_period = motion_scores[-self.celebration_duration//2:]
        high_motion = np.mean(celebration_period) > self.celebration_threshold
        
        # Detect goal if pattern matches
        if was_quiet and high_motion and current_motion > self.celebration_threshold:
            # The goal likely happened around the quiet-to-celebration transition
            goal_frame = frame_number - self.celebration_duration // 2
            goal_timestamp = self.get_timestamp(goal_frame)
            
            # Save the goal frame
            goal_frame_path = os.path.join(self.output_dir, f"goal_{len(self.detected_goals)+1}.jpg")
            cv2.imwrite(goal_frame_path, current_frame)
            
            # Update last goal frame
            self.last_goal_frame = frame_number
            
            # Record the goal
            self.detected_goals.append({
                "frame_number": goal_frame,
                "timestamp": goal_timestamp,
                "frame_path": goal_frame_path
            })
            
            return True
        
        return False

    def process_video(self):
        """Process the entire video and detect goals"""
        frame_number = 0
        
        print("Processing video...")
        start_time = time.time()
        
        # Process the video frame by frame
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Analyze the current frame
            processed_frame, motion_score = self.analyze_frame(frame, frame_number)
            
            # Add motion score to history
            self.motion_scores.append(motion_score)
            
            # Add frame to buffer
            self.frame_buffer.append(processed_frame)
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
            
            # Check for goal events
            is_goal = self.detect_goal_events(frame, self.motion_scores, frame_number)
            
            # Display progress
            if frame_number % 100 == 0:
                progress = (frame_number / self.frame_count) * 100
                print(f"Progress: {progress:.1f}% (Frame {frame_number}/{self.frame_count})", end="\r")
            
            # If goal was detected
            if is_goal:
                goal_info = self.detected_goals[-1]
                print(f"\nGOAL DETECTED at {goal_info['timestamp']} (Frame {goal_info['frame_number']})")
            
            # Show frame if in debug mode
            if self.debug:
                cv2.imshow('Goal Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_number += 1
        
        processing_time = time.time() - start_time
        print(f"\nVideo processing complete. Time taken: {processing_time:.2f} seconds")
        print(f"Detected {len(self.detected_goals)} goals")
        
        # Release resources
        self.cap.release()
        if self.debug:
            cv2.destroyAllWindows()
        
        return self.detected_goals

    def plot_motion_analysis(self):
        """Plot motion analysis and detected goals"""
        plt.figure(figsize=(15, 6))
        
        # Plot motion scores
        plt.plot(self.motion_scores, label='Motion Score', color='blue', alpha=0.7)
        
        # Plot goal events
        for goal in self.detected_goals:
            frame_num = goal['frame_number']
            plt.axvline(x=frame_num, color='red', linestyle='--', alpha=0.7)
            plt.text(frame_num, max(self.motion_scores) * 0.9, 
                     f"GOAL: {goal['timestamp']}", 
                     rotation=90, verticalalignment='top')
        
        # Plot thresholds
        plt.axhline(y=self.celebration_threshold, color='green', linestyle=':', label='Celebration Threshold')
        plt.axhline(y=self.quiet_threshold, color='orange', linestyle=':', label='Quiet Threshold')
        
        plt.title('Video Motion Analysis for Goal Detection')
        plt.xlabel('Frame Number')
        plt.ylabel('Motion Score (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, 'motion_analysis.png'), dpi=300, bbox_inches='tight')
        
        if self.debug:
            plt.show()

    def generate_report(self):
        """Generate a report of detected goals"""
        report_path = os.path.join(self.output_dir, 'goal_detection_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("===== Soccer Goal Detection Report =====\n\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Duration: {self.frame_count/self.fps:.2f} seconds\n")
            f.write(f"Resolution: {self.video_width}x{self.video_height}\n")
            f.write(f"FPS: {self.fps}\n\n")
            
            f.write(f"Total goals detected: {len(self.detected_goals)}\n\n")
            
            if self.detected_goals:
                f.write("Goal Timestamps:\n")
                for i, goal in enumerate(self.detected_goals, 1):
                    f.write(f"Goal #{i}: {goal['timestamp']} (Frame {goal['frame_number']})\n")
            else:
                f.write("No goals were detected in this video.\n")
        
        print(f"Report generated: {report_path}")
        return report_path

    def run(self):
        """Run the complete goal detection pipeline"""
        self.process_video()
        self.plot_motion_analysis()
        self.generate_report()
        return self.detected_goals

import yt_dlp

def download_youtube_video(youtube_url, output_path):
    try:
        print(f"Downloading video from: {youtube_url}")

        ydl_opts = {
            'outtmpl': output_path,  # e.g. '/path/to/save/%(title)s.%(ext)s'
            'format': 'best[ext=mp4]/best',
            'quiet': False,
            'noplaylist': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            downloaded_filename = ydl.prepare_filename(info_dict)
            print(f"Download complete: {downloaded_filename}")
            return downloaded_filename

    except Exception as e:
        print(f"Error downloading video: {e}")
        return output_path



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Soccer Goal Detection System')
    parser.add_argument('--video', type=str, default=None, help='Path to input video file')
    parser.add_argument('--youtube', type=str, default=None, help='YouTube URL to download and process')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Check input sources
    if args.youtube:
        video_path = download_youtube_video(args.youtube)
    elif args.video:
        video_path = args.video
    else:
        # Default to the video from the project description
        video_path = download_youtube_video('https://www.youtube.com/watch?v=nFg0N_JesWs', 'assets/soccer_goal.mp4')
    
    # Create and run the goal detection system
    detector = GoalDetectionSystem(
        video_path=video_path,
        output_dir=args.output,
        debug=args.debug
    )
    
    goals = detector.run()
    
    print("\nDetected Goals:")
    for i, goal in enumerate(goals, 1):
        print(f"Goal #{i}: {goal['timestamp']} (Frame {goal['frame_number']})")
    
    print(f"\nCheck {args.output} directory for detailed results")


if __name__ == "__main__":
    main()