import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

class ElbowAngleAnalyzer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (point2 is the vertex)"""
        # Convert to numpy arrays
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def get_elbow_angles(self, landmarks):
        """Calculate left and right elbow angles"""
        try:
            # Left arm landmarks
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            
            # Right arm landmarks  
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Calculate angles (elbow is the vertex)
            left_angle = None
            right_angle = None
            
            # Check visibility for left arm
            if (left_shoulder.visibility > 0.5 and 
                left_elbow.visibility > 0.5 and 
                left_wrist.visibility > 0.5):
                left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Check visibility for right arm
            if (right_shoulder.visibility > 0.5 and 
                right_elbow.visibility > 0.5 and 
                right_wrist.visibility > 0.5):
                right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
                
            return left_angle, right_angle
            
        except Exception as e:
            return None, None
    
    def process_frame(self, frame):
        """Process frame and return annotated frame with elbow angles"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        left_angle = None
        right_angle = None
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate elbow angles
            left_angle, right_angle = self.get_elbow_angles(results.pose_landmarks.landmark)
            
            # Draw angle text on frame
            h, w, _ = annotated_frame.shape
            
            if left_angle is not None:
                cv2.putText(annotated_frame, f'Left Elbow: {left_angle:.1f}°', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, 'Left Elbow: Not detected', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if right_angle is not None:
                cv2.putText(annotated_frame, f'Right Elbow: {right_angle:.1f}°', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, 'Right Elbow: Not detected', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), left_angle, right_angle
    
    def get_video_info(self, video_path):
        """Get video information"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return frame_count, fps, width, height
    
    def get_frame_at_position(self, video_path, frame_number):
        """Get specific frame from video"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

def main():
    st.set_page_config(
        page_title="Elbow Angle Analysis",
        page_icon="💪",
        layout="wide"
    )
    
    st.title("💪 Elbow Angle Analysis with MediaPipe")
    st.markdown("*Upload a video to analyze left and right elbow angles frame by frame*")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ElbowAngleAnalyzer()
    if 'video_loaded' not in st.session_state:
        st.session_state.video_loaded = False
    if 'angle_data' not in st.session_state:
        st.session_state.angle_data = {}
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to analyze elbow angles"
    )
    
    if uploaded_file is not None:
        # Save file temporarily
        if not st.session_state.video_loaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
                st.session_state.video_loaded = True
        
        try:
            # Get video information
            frame_count, fps, width, height = st.session_state.analyzer.get_video_info(st.session_state.video_path)
            
            # Display video information
            st.subheader("📹 Video Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("FPS", f"{fps:.1f}")
            with col2:
                st.metric("Total Frames", frame_count)
            with col3:
                st.metric("Duration", f"{frame_count/fps:.2f}s")
            with col4:
                st.metric("Resolution", f"{width}x{height}")
            
            # Frame navigation
            st.subheader("🎬 Frame Navigation")
            
            # Current frame selector
            current_frame = st.slider(
                "Current Frame",
                min_value=0,
                max_value=frame_count-1,
                value=0,
                help="Navigate through video frames"
            )
            
            # Navigation controls
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                if st.button("⏮️ Start"):
                    current_frame = 0
                    st.rerun()
            
            with col2:
                if st.button("⏪ -10"):
                    current_frame = max(0, current_frame - 10)
                    st.rerun()
            
            with col3:
                if st.button("⏪ -1"):
                    current_frame = max(0, current_frame - 1)
                    st.rerun()
            
            with col4:
                if st.button("⏩ +1"):
                    current_frame = min(frame_count-1, current_frame + 1)
                    st.rerun()
            
            with col5:
                if st.button("⏩ +10"):
                    current_frame = min(frame_count-1, current_frame + 10)
                    st.rerun()
            
            with col6:
                if st.button("⏭️ End"):
                    current_frame = frame_count-1
                    st.rerun()
            
            # Process current frame
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Check if frame is already processed
                if current_frame not in st.session_state.angle_data:
                    with st.spinner("Processing frame..."):
                        frame = st.session_state.analyzer.get_frame_at_position(st.session_state.video_path, current_frame)
                        
                        if frame is not None:
                            annotated_frame, left_angle, right_angle = st.session_state.analyzer.process_frame(frame)
                            
                            # Store processed data
                            st.session_state.angle_data[current_frame] = {
                                'annotated_frame': annotated_frame,
                                'left_angle': left_angle,
                                'right_angle': right_angle
                            }
                
                # Display frame
                if current_frame in st.session_state.angle_data:
                    frame_data = st.session_state.angle_data[current_frame]
                    current_time = current_frame / fps
                    
                    st.image(
                        frame_data['annotated_frame'],
                        caption=f"Frame {current_frame} | Time: {current_time:.3f}s",
                        use_column_width=True
                    )
            
            with col2:
                st.subheader("📐 Elbow Angles")
                
                # Current frame info
                current_time = current_frame / fps
                st.metric("Current Time", f"{current_time:.3f}s")
                st.metric("Frame Number", current_frame)
                
                # Display angles if available
                if current_frame in st.session_state.angle_data:
                    frame_data = st.session_state.angle_data[current_frame]
                    
                    # Left elbow angle
                    if frame_data['left_angle'] is not None:
                        st.metric(
                            "Left Elbow", 
                            f"{frame_data['left_angle']:.1f}°",
                            help="Angle between left shoulder-elbow-wrist"
                        )
                    else:
                        st.warning("Left elbow not detected")
                    
                    # Right elbow angle
                    if frame_data['right_angle'] is not None:
                        st.metric(
                            "Right Elbow", 
                            f"{frame_data['right_angle']:.1f}°",
                            help="Angle between right shoulder-elbow-wrist"
                        )
                    else:
                        st.warning("Right elbow not detected")
                
                # Angle interpretation
                st.subheader("📊 Interpretation")
                st.markdown("""
                **Elbow Angle Guidelines:**
                - **180°**: Fully extended arm
                - **90°**: Right angle bend
                - **< 90°**: Deep flexion
                - **> 90°**: Partial extension
                
                *Angles closer to 0° indicate maximum flexion*
                """)
                
                # Batch processing option
                st.subheader("⚡ Batch Processing")
                if st.button("🔄 Process All Frames", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process every 10th frame for speed
                    step = max(1, frame_count // 100)
                    
                    for i in range(0, frame_count, step):
                        if i not in st.session_state.angle_data:
                            frame = st.session_state.analyzer.get_frame_at_position(st.session_state.video_path, i)
                            
                            if frame is not None:
                                annotated_frame, left_angle, right_angle = st.session_state.analyzer.process_frame(frame)
                                
                                st.session_state.angle_data[i] = {
                                    'annotated_frame': annotated_frame,
                                    'left_angle': left_angle,
                                    'right_angle': right_angle
                                }
                        
                        progress = i / frame_count
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {i}/{frame_count}")
                    
                    progress_bar.progress(1.0)
                    st.success("✅ Batch processing complete!")
                
                # Export data
                if st.session_state.angle_data and st.button("📊 Export Angle Data", use_container_width=True):
                    # Create export data
                    export_data = []
                    for frame_num in sorted(st.session_state.angle_data.keys()):
                        data = st.session_state.angle_data[frame_num]
                        export_data.append({
                            'frame': frame_num,
                            'time_seconds': frame_num / fps,
                            'left_elbow_angle': data['left_angle'],
                            'right_elbow_angle': data['right_angle']
                        })
                    
                    # Display as text for copying
                    export_text = "Frame,Time(s),Left_Elbow,Right_Elbow\n"
                    for data in export_data:
                        left_angle = f"{data['left_elbow_angle']:.1f}" if data['left_elbow_angle'] is not None else "N/A"
                        right_angle = f"{data['right_elbow_angle']:.1f}" if data['right_elbow_angle'] is not None else "N/A"
                        export_text += f"{data['frame']},{data['time_seconds']:.3f},{left_angle},{right_angle}\n"
                    
                    st.code(export_text, language="csv")
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        
        finally:
            # Clean up temp file when done
            pass
    
    else:
        # Instructions when no video loaded
        st.markdown("""
        ### 📋 How to Use:
        
        1. **📤 Upload Video**: Choose any video file with visible arm movements
        2. **🎬 Navigate Frames**: Use slider and controls to move through the video
        3. **📐 View Angles**: See real-time left and right elbow angles
        4. **⚡ Batch Process**: Analyze all frames for complete dataset
        5. **📊 Export Data**: Copy angle measurements for further analysis
        
        ### 🎯 Best Results:
        - **Clear view** of both arms
        - **Good lighting** and contrast
        - **Side or front view** of subject
        - **Visible shoulder, elbow, wrist** landmarks
        
        ### 📐 Angle Calculation:
        - Uses **shoulder-elbow-wrist** landmarks
        - **MediaPipe pose estimation** for accurate tracking
        - **Real-time processing** with visual feedback
        - **180° = fully extended**, **0° = maximum flexion**
        
        ### 💡 Applications:
        - **Exercise form analysis**
        - **Physical therapy assessment**
        - **Sports biomechanics**
        - **Movement pattern analysis**
        """)

if __name__ == "__main__":
    main()
