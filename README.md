# rim_examples
Example implementations using Robotics Ipc Module header only libraries

## Examples

### Multi-Process Video Classifier Dashboard

A multi-process machine learning demo using shared memory IPC:

- **Process 0** (C++): Message router managing SHM channels
- **Process 1** (Python): Video publisher streaming from online sources
- **Process 2** (C++ / CUDA): Real-time YOLO object detection
- **Process 3** (Node.js): Web-based dashboard with live video and stats

➡️ [examples/multi_process/video_classifier_dashboard](examples/multi_process/video_classifier_dashboard/)
