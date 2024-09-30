from ultralytics import YOLO
import cv2
from tqdm import tqdm


class ObjectDetector:
    """
    A class for object detection which uses YOLO model for detecting objects in a video.

    Attributes:
        model (str): The preferred YOLO model to load as: YOLO(model), as defined in ultralytics docs.
        source (str): Path to the input video file.
        output_path (str): Path to save the output video.
        percentage_to_process (int): Percentage of the video frames to process.
    """
    def __init__(self, source, output_path='output_video.mp4', model='yolov8m-seg.pt', percentage_to_process=100):
        """Initializes the ObjectDetector."""
        self.source = source
        self.output_path = output_path
        self.model = YOLO(model)
        self.percentage_to_process = percentage_to_process

        self.cap = None
        self.out = None
        self.total_frames = 0
        self.frames_to_process = 0

    def initialize_video(self):
        """
        Initializes video capture and output video writer.
        """
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError("Error opening video source")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames_to_process = int((self.percentage_to_process / 100) * self.total_frames)

    def process_frames(self):
        """Processes the specified percentage of frames from the video."""
        processed_frames = 0
        with tqdm(total=self.frames_to_process, desc="Processing Video", unit="frame") as pbar:
            while self.cap.isOpened() and processed_frames < self.frames_to_process:
                ret, frame = self.cap.read()
                if not ret:
                    break

                results = self.model(frame, verbose=False)
                frame = results[0].plot()

                self.out.write(frame)

                processed_frames += 1
                pbar.update(1)

    def release_resources(self):
        """Releases all resources used by the object detector."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

    def run(self):
        """Runs the whole object detection process."""
        try:
            self.initialize_video()
            self.process_frames()
        finally:
            self.release_resources()


# detector = ObjectDetector("desk.mp4", "output_video.mp4","yolov8m-seg.pt")
# detector.run()
