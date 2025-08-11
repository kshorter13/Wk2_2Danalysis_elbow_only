import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# Configure Streamlit
st.set_page_config(
    page_title="Elbow Angle Analysis",
    page_icon="💪",
    layout="wide"
)

class ElbowAngleAnalyzer:
    def __init__(self):
        try:
            # Initialize MediaPipe (0.9.1 compatible)
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Initialize pose detection with 0.9.1 compatible settings
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.initialized = True
            
        except Exception as e:
            st.error(f"Failed to initialize MediaPipe: {str(e)}")
            self.initialized = False
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (point2 is the vertex)"""
        try:
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
        except Exception as e:
            return None
    
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
        if not self.initialized:
            return frame, None, None
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            left_angle = None
            right_angle = None
            
            if results.pose_landmarks:
                # Draw pose landmarks (0.9.1 compatible)
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
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
            
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            return frame, None, None
    
    def get_video_info(self, video_path):
        """Get video information"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            return frame_count, fps, width, height
        except Exception as e:
            st.error(f"Error getting video info: {str(e)}")
            return 0, 0, 0, 0
    
    def get_frame_at_position(self, video_path, frame_number):
        """Get specific frame from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        except Exception as e:
            st.error(f"Error getting frame: {str(e)}")
            return None

def main():
    st.title("💪 Elbow Angle Analysis with MediaPipe")
    st.markdown("*Upload a video to analyze left and right elbow angles frame by frame*")
    
    # Show version info
    with st.expander("📊 System Information"):
        st.info(f"OpenCV Version: {cv2.__version__}")
        st.info(f"MediaPipe Version: {mp.__version__}")
        st.info(f"NumPy Version: {np.__version__}")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        with st.spinner("Initializing MediaPipe..."):
            st.session_state.analyzer = ElbowAngleAnalyzer()
    
    if 'video_loaded' not in st.session_state:
        st.session_state.video_loaded = False
    if 'angle_data' not in st.session_state:
        st.session_state.angle_data = {}
    if 'current_frame_to_analyze' not in st.session_state:
        st.session_state.current_frame_to_analyze = 0
    
    # Check if analyzer initialized properly
    if not st.session_state.analyzer.initialized:
        st.error("❌ MediaPipe failed to initialize. Please try refreshing the page.")
        st.info("💡 If the problem persists, check the logs for more details.")
        return
    else:
        st.success("✅ MediaPipe initialized successfully!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Video File",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to analyze elbow angles"
    )
    
    if uploaded_file is not None:
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 100:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f}MB). Processing may be slow.")
        
        # Save file temporarily
        if not st.session_state.video_loaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
                st.session_state.video_loaded = True
        
        try:
            # Get video information
            frame_count, fps, width, height = st.session_state.analyzer.get_video_info(st.session_state.video_path)
            
            if frame_count == 0:
                st.error("❌ Could not read video file. Please try a different file.")
                return
            
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
            
            # Performance warning for large videos
            if frame_count > 1000:
                st.warning(f"⚠️ Large video detected ({frame_count} frames). Be selective with analysis.")
            
            # Frame navigation
            st.subheader("🎬 Frame Navigation & Analysis")
            
            # Current frame selector for navigation
            current_frame_to_display = st.slider(
                "Select Frame to View",
                min_value=0,
                max_value=frame_count-1,
                value=st.session_state.current_frame_to_analyze,
                help="Navigate through video frames without processing"
            )
            st.session_state.current_frame_to_analyze = current_frame_to_display
            
            # Navigation controls
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                if st.button("⏮️ Start"):
                    st.session_state.current_frame_to_analyze = 0
                    st.rerun()
            
            with col2:
                if st.button("⏪ -10"):
                    st.session_state.current_frame_to_analyze = max(0, st.session_state.current_frame_to_analyze - 10)
                    st.rerun()
            
            with col3:
                if st.button("⏪ -1"):
                    st.session_state.current_frame_to_analyze = max(0, st.session_state.current_frame_to_analyze - 1)
                    st.rerun()
            
            with col4:
                if st.button("⏩ +1"):
                    st.session_state.current_frame_to_analyze = min(frame_count-1, st.session_state.current_frame_to_analyze + 1)
                    st.rerun()
            
            with col5:
                if st.button("⏩ +10"):
                    st.session_state.current_frame_to_analyze = min(frame_count-1, st.session_state.current_frame_to_analyze + 10)
                    st.rerun()
            
            with col6:
                if st.button("⏭️ End"):
                    st.session_state.current_frame_to_analyze = frame_count-1
                    st.rerun()

            # Button to trigger analysis for the currently selected frame
            if st.button("📐 Analyze This Frame", use_container_width=True):
                # Check if frame is already processed
                if current_frame_to_display not in st.session_state.angle_data:
                    with st.spinner(f"Processing frame {current_frame_to_display}..."):
                        frame_to_process = st.session_state.analyzer.get_frame_at_position(st.session_state.video_path, current_frame_to_display)
                        
                        if frame_to_process is not None:
                            annotated_frame, left_angle, right_angle = st.session_state.analyzer.process_frame(frame_to_process)
                            
                            # Store processed data in the session state cache
                            st.session_state.angle_data[current_frame_to_display] = {
                                'annotated_frame': annotated_frame,
                                'left_angle': left_angle,
                                'right_angle': right_angle
                            }
                        else:
                            st.error(f"Could not read frame {current_frame_to_display}")
                    st.success(f"✅ Frame {current_frame_to_display} analyzed and angles stored!")
            
            # --- Display section ---
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Display the video frame
                frame_to_display = st.session_state.analyzer.get_frame_at_position(st.session_state.video_path, current_frame_to_display)
                if frame_to_display is not None:
                    # Check if the frame has been processed and display the annotated version
                    if current_frame_to_display in st.session_state.angle_data:
                        annotated_frame = st.session_state.angle_data[current_frame_to_display]['annotated_frame']
                    else:
                        # Otherwise, just show the raw frame
                        annotated_frame = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)

                    current_time = current_frame_to_display / fps if fps > 0 else 0
                    st.image(
                        annotated_frame,
                        caption=f"Frame {current_frame_to_display} | Time: {current_time:.3f}s",
                        use_column_width=True
                    )
                
            with col2:
                st.subheader("📐 Elbow Angles")
                
                # Current frame info
                current_time = current_frame_to_display / fps if fps > 0 else 0
                st.metric("Current Time", f"{current_time:.3f}s")
                st.metric("Frame Number", current_frame_to_display)
                
                # Display angles if available from the cache
                if current_frame_to_display in st.session_state.angle_data:
                    frame_data = st.session_state.angle_data[current_frame_to_display]
                    
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
                else:
                    st.info("Click 'Analyze This Frame' to see angles.")
                
                # Angle interpretation guide
                with st.expander("📊 Angle Guide"):
                    st.markdown("""
                    **Elbow Angle Guidelines:**
                    - **180°**: Fully extended arm
                    - **90°**: Right angle bend
                    - **< 90°**: Deep flexion
                    - **> 90°**: Partial extension
                    
                    *Angles closer to 0° indicate maximum flexion*
                    """)
                
            # Button to clear the analysis cache
            if st.button("🗑️ Clear Analysis Cache", use_container_width=True):
                st.session_state.angle_data = {}
                st.success("✅ Analysis cache cleared!")

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("💡 Try uploading a different video file or refresh the page.")
    
    else:
        # Instructions when no video loaded
        st.markdown("""
        ### 📋 How to Use:
        
        1. **📤 Upload Video**: Choose any video file with visible arm movements
        2. **🎬 Navigate Frames**: Use the slider and controls to move through the video
        3. **📐 Analyze Frames**: Click **Analyze This Frame** to get elbow angles
        
        ### 🎯 Best Results:
        - **Clear view** of both arms in the video
        - **Good lighting** and contrast
        - **Side or front view** of the subject
        - **Visible shoulder, elbow, wrist** landmarks
        - **Videos under 2 minutes** for optimal performance
        
        ### 📐 Angle Calculation:
        - Uses **shoulder-elbow-wrist** landmarks
        - **MediaPipe pose estimation** for accurate tracking
        - **Real-time processing** with visual feedback
        - **180° = fully extended**, **0° = maximum flexion**
        """)

if __name__ == "__main__":
    main()
