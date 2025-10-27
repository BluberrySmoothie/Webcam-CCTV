import cv2
import os
import time
import shutil
import glob
import threading
import numpy as np
from queue import Queue
from datetime import datetime
import ffmpeg
import platform
import subprocess

# --- Configuration ---
PHOTO_INTERVAL = 0.1  # Take a photo every 0.1 seconds
RECORD_DURATION_MINUTES = 1
RECORD_DIR = "recordings"
TEMP_PHOTOS_DIR = "temp_photos"
DISK_SPACE_THRESHOLD_GB = 5

# --- Camera reconnection settings ---
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2  # seconds between reconnection attempts
CAMERA_INDEX = 1  # Default, will be set by user selection

# --- Auto-brightness settings ---
AUTO_BRIGHTNESS_ENABLED = True
TARGET_BRIGHTNESS = 120  # Target average brightness (0-255)
BRIGHTNESS_TOLERANCE = 15  # How much deviation to allow before adjusting
BRIGHTNESS_CHECK_INTERVAL = 5  # Check brightness every N photos
BRIGHTNESS_ADJUSTMENT_STEP = 25  # How much to adjust brightness each time

def get_camera_names():
    """Get camera names from the system."""
    system = platform.system()
    camera_names = {}
    
    try:
        if system == "Windows":
            # Use Windows Registry to get camera names
            try:
                result = subprocess.run(
                    ["powershell", "-Command", 
                     "Get-WmiObject Win32_PnPEntity -Filter \"Name LIKE '%camera%' OR Name LIKE '%webcam%' OR Name LIKE '%video%'\" | Select-Object -ExpandProperty Name"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                    for idx, line in enumerate(lines):
                        camera_names[idx] = line
            except:
                pass
            
            # Fallback: Try alternative method using device manager
            if not camera_names:
                try:
                    result = subprocess.run(
                        ["powershell", "-Command", 
                         "$devices = Get-PnpDevice -Class Camera -ErrorAction SilentlyContinue; $devices | ForEach-Object {$_.FriendlyName}"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                        for idx, line in enumerate(lines):
                            camera_names[idx] = line
                except:
                    pass
        
        elif system == "Linux":
            # Use v4l2-ctl or check /sys/class/video4linux
            try:
                result = subprocess.run(
                    ["ls", "-1", "/sys/class/video4linux/"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    devices = result.stdout.strip().split('\n')
                    for idx, device in enumerate(devices):
                        if device:
                            try:
                                name_path = f"/sys/class/video4linux/{device}/name"
                                if os.path.exists(name_path):
                                    with open(name_path, 'r') as f:
                                        camera_names[idx] = f.read().strip()
                            except:
                                pass
            except:
                pass
        
        elif system == "Darwin":
            # macOS: Use system_profiler
            try:
                result = subprocess.run(
                    ["system_profiler", "SPCameraDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    idx = 0
                    for line in lines:
                        if "Model" in line or "Internal" in line:
                            camera_names[idx] = line.split(':', 1)[-1].strip()
                            idx += 1
            except:
                pass
    
    except Exception:
        pass
    
    return camera_names

# --- Video settings ---
frame_width = 640
frame_height = 480
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# --- Global queue for background video creation ---
video_creation_queue = Queue()

# --- Global variables for brightness control ---
last_brightness_check = 0
current_brightness = 0
current_exposure = 0

def detect_available_cameras(max_index=10):
    """Detect all available cameras on the system."""
    available_cameras = []
    
    print("Scanning for available cameras...")
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # Try to read a frame to verify it's a working camera
            ret, frame = cap.read()
            if ret:
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
                
                print(f"✓ Found camera at index {i}: {width}x{height} @ {fps} FPS")
            
            cap.release()
    
    return available_cameras

def select_camera():
    """Display available cameras and let user select one."""
    available_cameras = detect_available_cameras()
    camera_names = get_camera_names()
    
    if not available_cameras:
        print("✗ No cameras found on this system!")
        return None
    
    print("\n" + "=" * 60)
    print("AVAILABLE CAMERAS")
    print("=" * 60)
    
    for idx, camera in enumerate(available_cameras, 1):
        index = camera['index']
        name = camera_names.get(index - 1, "Unknown Camera")
        resolution = f"{camera['width']}x{camera['height']}"
        fps = camera['fps']
        
        print(f"\n{idx}. {name}")
        print(f"   Index: {index}")
        print(f"   Resolution: {resolution}")
        print(f"   FPS: {fps}")
    
    print("\n" + "=" * 60)
    
    while True:
        try:
            selection = input(f"\nSelect a camera (1-{len(available_cameras)}): ").strip()
            choice = int(selection)
            
            if 1 <= choice <= len(available_cameras):
                selected_camera = available_cameras[choice - 1]
                print(f"\n✓ Selected camera at index {selected_camera['index']}")
                return selected_camera['index']
            else:
                print(f"Please enter a number between 1 and {len(available_cameras)}")
        except ValueError:
            print("Please enter a valid number")

def get_image_brightness(frame):
    """Calculate the average brightness of an image."""
    # Convert to grayscale and calculate mean
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def adjust_camera_settings(cap, frame):
    """Automatically adjust camera brightness/exposure based on image content."""
    global last_brightness_check, current_brightness, current_exposure
    
    if not AUTO_BRIGHTNESS_ENABLED:
        return
    
    # Only check brightness every N photos to avoid constant adjustments
    last_brightness_check += 1
    if last_brightness_check < BRIGHTNESS_CHECK_INTERVAL:
        return
    
    last_brightness_check = 0
    
    # Calculate current image brightness
    avg_brightness = get_image_brightness(frame)
    
    print(f"\nBrightness check: {avg_brightness:.1f} (target: {TARGET_BRIGHTNESS})")
    
    # Determine if adjustment is needed
    brightness_diff = TARGET_BRIGHTNESS - avg_brightness
    
    if abs(brightness_diff) < BRIGHTNESS_TOLERANCE:
        print("✓ Brightness is within acceptable range")
        return
    
    # Try to adjust exposure first (usually more effective)
    if cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1:  # If auto-exposure is on
        # Turn off auto-exposure to allow manual control
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
        time.sleep(0.1)  # Give camera time to adjust
    
    # Get current exposure value
    current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    
    if brightness_diff > 0:  # Image too dark
        # Increase exposure (make brighter)
        new_exposure = current_exposure + 1
        cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
        print(f"📈 Increasing exposure: {current_exposure:.1f} → {new_exposure:.1f}")
    else:  # Image too bright
        # Decrease exposure (make darker)
        new_exposure = current_exposure - 1
        cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
        print(f"📉 Decreasing exposure: {current_exposure:.1f} → {new_exposure:.1f}")
    
    # If exposure adjustment doesn't work well, try brightness
    if abs(brightness_diff) > BRIGHTNESS_TOLERANCE * 2:
        current_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        
        if brightness_diff > 0:  # Still too dark
            new_brightness = min(current_brightness + BRIGHTNESS_ADJUSTMENT_STEP, 255)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
            print(f"💡 Adjusting brightness: {current_brightness:.0f} → {new_brightness:.0f}")
        else:  # Still too bright
            new_brightness = max(current_brightness - BRIGHTNESS_ADJUSTMENT_STEP, 0)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
            print(f"🔆 Adjusting brightness: {current_brightness:.0f} → {new_brightness:.0f}")

def get_disk_space(path):
    """Returns available disk space in GB."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

def delete_oldest_file(directory):
    """Deletes the oldest video file in the specified directory."""
    files = sorted(glob.glob(os.path.join(directory, "*.mp4")), key=os.path.getmtime)
    if files:
        oldest_file = files[0]
        print(f"Deleting oldest file to free up space: {oldest_file}")
        os.remove(oldest_file)

def create_video_from_photos(photos_dir, output_path):
    """Create a Variable Frame Rate (VFR) video from a sequence of photos using ffmpeg."""
    try:
        photo_pattern = os.path.join(photos_dir, "*.jpg")
        photos = sorted(glob.glob(photo_pattern))
        
        if not photos:
            print(f"No photos found in {photos_dir}")
            return False
            
        print(f"Creating VFR video from {len(photos)} photos...")
        
        photo_data = []
        for photo_path in photos:
            timestamp_str = os.path.basename(photo_path).split('.')[0]
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
            photo_data.append((photo_path, timestamp))
            
        # Create a temporary file with the full absolute paths
        timestamp_file_path = os.path.join(TEMP_PHOTOS_DIR, 'timestamps.txt')
        with open(timestamp_file_path, 'w') as f:
            for i in range(len(photos)):
                f.write(f"file '{os.path.abspath(photos[i])}'\n")
                if i < len(photos) - 1:
                    duration = (photo_data[i+1][1] - photo_data[i][1]).total_seconds()
                    f.write(f"duration {duration}\n")
        
        # Use FFmpeg's concat demuxer to create the video
        ffmpeg_command = (
            ffmpeg
            .input(timestamp_file_path, format='concat', safe=0)
            .output(output_path, vcodec='libx264', pix_fmt='yuv420p', preset='fast')
        )
        
        ffmpeg_command.run(overwrite_output=True)
        
        # Clean up temporary files
        os.remove(timestamp_file_path)
        shutil.rmtree(photos_dir)
        
        print(f"\n✓ VFR video created: {os.path.basename(output_path)} ({len(photos)} frames)")
        return True
        
    except ffmpeg.Error as e:
        print(f"Error creating video with ffmpeg: {e.stderr.decode('utf8')}")
        return False
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def background_video_creator():
    """Background thread that creates videos from photos."""
    while True:
        try:
            task = video_creation_queue.get(timeout=1)
            if task is None:  # Stop signal
                break
            
            photos_session_dir, output_path, _ = task
            create_video_from_photos(photos_session_dir, output_path)
            video_creation_queue.task_done()
            
        except:
            continue

def initialize_camera_settings(cap):
    """Initialize camera with optimal settings for auto-brightness."""
    print("Initializing camera settings...")
    
    # Set basic properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for fresher frames
    
    if AUTO_BRIGHTNESS_ENABLED:
        # Get current settings
        auto_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        contrast = cap.get(cv2.CAP_PROP_CONTRAST)
        
        print(f"Initial camera settings:")
        print(f"  Auto-exposure: {auto_exposure}")
        print(f"  Brightness: {brightness}")
        print(f"  Exposure: {exposure}")
        print(f"  Contrast: {contrast}")
        
        # Set reasonable starting values
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # Middle value
        cap.set(cv2.CAP_PROP_CONTRAST, 128)    # Middle value
        
        print("Auto-brightness enabled")
    else:
        print("Auto-brightness disabled - using manual settings")

def reconnect_camera(camera_index):
    """Attempt to reconnect to the camera with retries."""
    print(f"\n⚠️  Camera disconnected! Attempting to reconnect...")
    
    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
        print(f"Reconnection attempt {attempt}/{MAX_RECONNECT_ATTEMPTS}...")
        
        # Release any existing capture object
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Camera reconnected successfully on attempt {attempt}!")
                    initialize_camera_settings(cap)
                    return cap
                else:
                    cap.release()
            else:
                cap.release()
        except Exception as e:
            print(f"Reconnection error: {e}")
        
        if attempt < MAX_RECONNECT_ATTEMPTS:
            print(f"Waiting {RECONNECT_DELAY} seconds before next attempt...")
            time.sleep(RECONNECT_DELAY)
    
    print(f"✗ Failed to reconnect after {MAX_RECONNECT_ATTEMPTS} attempts")
    return None

def is_camera_healthy(cap):
    """Check if the camera is still functioning properly."""
    if not cap.isOpened():
        return False
    
    # Try to read a frame to verify camera is working
    ret, frame = cap.read()
    return ret

def start_recording(camera_index):
    global last_brightness_check
    
    # Create directories
    for directory in [RECORD_DIR, TEMP_PHOTOS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}.")
        return
    
    # Initialize camera settings
    initialize_camera_settings(cap)
    
    print(f"Taking photos every {PHOTO_INTERVAL}s, creating VFR videos")
    if AUTO_BRIGHTNESS_ENABLED:
        print(f"Auto-brightness: Target={TARGET_BRIGHTNESS}, Tolerance=±{BRIGHTNESS_TOLERANCE}")
    print(f"Auto-reconnect enabled: Max {MAX_RECONNECT_ATTEMPTS} attempts with {RECONNECT_DELAY}s delay")
    
    video_thread = threading.Thread(target=background_video_creator, daemon=True)
    video_thread.start()
    print("Background video creation thread started")
    
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3
    
    try:
        while True:
            if get_disk_space(RECORD_DIR) < DISK_SPACE_THRESHOLD_GB:
                print("Warning: Low disk space detected.")
                while get_disk_space(RECORD_DIR) < DISK_SPACE_THRESHOLD_GB:
                    delete_oldest_file(RECORD_DIR)
                    if not glob.glob(os.path.join(RECORD_DIR, "*.mp4")):
                        print("No more videos to delete. Stopping recording.")
                        return
            
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_photos_dir = os.path.join(TEMP_PHOTOS_DIR, f"session_{session_timestamp}")
            final_video_path = os.path.join(RECORD_DIR, f"{session_timestamp}.mp4")
            
            os.makedirs(session_photos_dir, exist_ok=True)
            
            print(f"Starting photo session: {session_timestamp}")
            
            total_duration = RECORD_DURATION_MINUTES * 60
            
            start_time = time.time()
            photos_taken = 0
            last_brightness_check = 0  # Reset brightness check counter
            
            next_photo_time = start_time
            
            while time.time() - start_time < total_duration:
                current_time = time.time()
                
                if current_time >= next_photo_time:
                    # Check camera health before attempting to read
                    if not is_camera_healthy(cap):
                        consecutive_failures += 1
                        print(f"\n⚠️  Camera read failed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                        
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            print("Multiple consecutive failures detected. Attempting reconnection...")
                            cap.release()
                            cap = reconnect_camera(camera_index)
                            
                            if cap is None:
                                print("✗ Unable to reconnect to camera. Exiting...")
                                return
                            
                            consecutive_failures = 0
                            last_brightness_check = 0
                            continue
                        else:
                            # Wait a bit and retry
                            time.sleep(0.5)
                            continue
                    
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_failures += 1
                        print(f"\n⚠️  Failed to read frame ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                        
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            print("Multiple consecutive failures detected. Attempting reconnection...")
                            cap.release()
                            cap = reconnect_camera(camera_index)
                            
                            if cap is None:
                                print("✗ Unable to reconnect to camera. Exiting...")
                                return
                            
                            consecutive_failures = 0
                            last_brightness_check = 0
                            continue
                        else:
                            time.sleep(0.5)
                            continue
                    
                    # Reset failure counter on successful read
                    consecutive_failures = 0
                    
                    # Auto-adjust brightness based on current frame
                    adjust_camera_settings(cap, frame)
                    
                    precise_time = datetime.now()
                    
                    # Add brightness info to timestamp if auto-brightness is enabled
                    if AUTO_BRIGHTNESS_ENABLED:
                        avg_brightness = get_image_brightness(frame)
                        text = f"{precise_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | B:{avg_brightness:.0f}"
                    else:
                        text = precise_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size, _ = cv2.getTextSize(text, font, 0.5, 1)
                    text_w, text_h = text_size
                    x, y = (frame_width - text_w - 5, frame_height - 5)
                    
                    cv2.rectangle(frame, (x - 2, y - text_h - 2), (x + text_w + 2, y + 2), (0, 0, 0), -1)
                    cv2.putText(frame, text, (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    photo_filename = precise_time.strftime("%Y%m%d_%H%M%S_%f.jpg")
                    photo_path = os.path.join(session_photos_dir, photo_filename)
                    
                    cv2.imwrite(photo_path, frame)
                    
                    photos_taken += 1
                    next_photo_time += PHOTO_INTERVAL
                    
                    elapsed = current_time - start_time
                    print(f"Taking photos... {elapsed:.1f}/{total_duration}s ({photos_taken} photos)", end='\r')
                else:
                    time_to_sleep = next_photo_time - current_time
                    if time_to_sleep > 0:
                        time.sleep(min(time_to_sleep, 0.1))  # Check more frequently for camera issues
            
            actual_duration = time.time() - start_time
            print(f"\nPhoto session complete: {actual_duration:.1f}s, {photos_taken} photos")
            
            # Only queue video creation if we have photos
            if photos_taken > 0:
                video_creation_queue.put((session_photos_dir, final_video_path, None))
                print(f"Queued for video creation: {session_timestamp}.mp4")
            else:
                print(f"No photos taken, skipping video creation")
                # Clean up empty session directory
                if os.path.exists(session_photos_dir):
                    shutil.rmtree(session_photos_dir)
            
            print("-" * 50)
    
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        video_creation_queue.put(None)
        video_thread.join(timeout=10)
        
        print("Recording stopped.")

if __name__ == "__main__":
    camera_index = select_camera()
    
    if camera_index is not None:
        start_recording(camera_index)
    else:
        print("No camera selected. Exiting.")