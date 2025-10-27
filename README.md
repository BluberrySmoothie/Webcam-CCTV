# WebCam Recorder

A Python-based CCTV recording application that automatically captures photos at regular intervals and converts them into variable frame rate (VFR) videos. Features automatic brightness adjustment, camera reconnection, and intelligent disk space management.

## Support the Project
If you like my work, feel free to buy me a coffee! ☕
https://buymeacoffee.com/BluberrySmoothie

## Features

- **Multi-Camera Support**: Automatically detects and lists all available cameras on your system
- **Camera Selection**: Interactive menu to choose which camera to use
- **Auto-Brightness Adjustment**: Automatically adjusts exposure and brightness to maintain target brightness levels
- **Variable Frame Rate (VFR) Video**: Creates smooth videos from photo sequences, maintaining actual capture timing
- **Camera Reconnection**: Automatically attempts to reconnect if the camera disconnects
- **Disk Space Management**: Automatically deletes oldest videos when disk space falls below threshold
- **Background Processing**: Creates videos in the background while recording continues
- **Timestamp Overlay**: Adds precise timestamps and brightness information to each frame

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- FFmpeg
- NumPy

### System-Specific Requirements

**Windows**: PowerShell (usually pre-installed)

**Linux**: `v4l2-ctl` (optional, for better camera name detection)

**macOS**: Built-in `system_profiler`

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/webcam-recorder.git
   cd webcam-recorder
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install opencv-python numpy ffmpeg-python
   ```

4. **Install FFmpeg**
   
   **Windows** (using Chocolatey):
   ```bash
   choco install ffmpeg
   ```
   
   **macOS** (using Homebrew):
   ```bash
   brew install ffmpeg
   ```
   
   **Linux** (Debian/Ubuntu):
   ```bash
   sudo apt-get install ffmpeg
   ```

## Usage

Run the script:
```bash
python WebCam.py
```

On startup, the application will:
1. Scan for available cameras
2. Display a list with camera names, resolutions, and frame rates
3. Prompt you to select which camera to use
4. Begin recording in 1-minute sessions
5. Convert each session to an MP4 video file

### Configuration

Edit the settings at the top of `WebCam.py`:

```python
PHOTO_INTERVAL = 0.1                    # Take a photo every 0.1 seconds
RECORD_DURATION_MINUTES = 1             # Duration of each recording session
RECORD_DIR = "recordings"               # Directory to save videos
DISK_SPACE_THRESHOLD_GB = 5             # Minimum free disk space before cleanup

# Auto-brightness settings
AUTO_BRIGHTNESS_ENABLED = True
TARGET_BRIGHTNESS = 120                 # Target average brightness (0-255)
BRIGHTNESS_TOLERANCE = 15               # Acceptable deviation
BRIGHTNESS_CHECK_INTERVAL = 5           # Check brightness every N photos
BRIGHTNESS_ADJUSTMENT_STEP = 25         # Adjustment amount per step

# Camera reconnection
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2                     # Seconds between reconnection attempts
```

## Output

Videos are saved to the `recordings/` directory with timestamps:
- Format: `YYYYMMDD_HHMMSS.mp4`
- Codec: H.264 (libx264)
- Resolution: 640x480 (configurable)

Temporary photo files are stored in `temp_photos/` and automatically cleaned up after video creation.

## Features in Detail

### Auto-Brightness Adjustment

The application monitors frame brightness and automatically adjusts camera exposure and brightness settings to maintain a target brightness level. This is useful for:
- Varying lighting conditions
- Long recording sessions with changing light
- Ensuring consistent video quality

### Automatic Reconnection

If the camera disconnects during recording:
1. The application detects the disconnection
2. Automatically attempts to reconnect (up to 5 times by default)
3. Resumes recording if successful
4. Exits gracefully if reconnection fails

### Disk Space Management

When available disk space drops below the threshold:
1. The application pauses recording
2. Automatically deletes the oldest video files
3. Continues recording once space is freed up

### Background Video Creation

Videos are created in a background thread to avoid interrupting the photo capture process, ensuring consistent photo intervals even during video encoding.

## Troubleshooting

**Cameras showing as "Unknown Camera"**
- This is normal on some systems
- The application will still work correctly
- Camera names depend on driver installation and Windows Device Manager detection

**"Could not open webcam" error**
- Check that your camera is connected and working
- Try a different camera index if multiple are available
- Verify camera is not in use by another application

**FFmpeg errors**
- Ensure FFmpeg is installed and in your system PATH
- Test with: `ffmpeg -version`

**Low video quality**
- Adjust `frame_width` and `frame_height` settings
- Enable `AUTO_BRIGHTNESS_ENABLED` for better lighting
- Check lighting conditions in your recording environment

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## Acknowledgments

- Built with [OpenCV](https://opencv.org/) for camera capture
- Uses [FFmpeg](https://ffmpeg.org/) for video encoding
- Cross-platform camera detection using system tools
