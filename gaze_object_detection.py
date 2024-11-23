import logging
from plugin import Plugin
import cv2
import numpy as np
from ultralytics import YOLO

CUSTOM_TOPIC = "gaze_object"

def get_object_in_gaze(gaze, r):
    for result in r:
        boxes = result.boxes
        classes = boxes.cls
        for i, box in enumerate(boxes.xyxyn):
            # Check if the gaze point lies within the bounding box of detected objects
            if box[0] <= gaze[0] <= box[2] and box[1] <= (1 - gaze[1]) <= box[3]:
                return result.names[int(classes[i])]
    return None

class GazeObjectDetector(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.origin_distance_tr = 0.1
        
        self.gaze_buffer = []
        self.curr_fixation = None
        self.curr_object = None
        
        self.model = YOLO('yolo11n.pt')
        
    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Gaze Object Detector"
        
        
    def _gaze_in_origin(self, gaze):
        return gaze["norm_pos"][0] < self.origin_distance_tr and gaze["norm_pos"][1] < self.origin_distance_tr
        
    def recent_events(self, events):
        fixations = events.get("fixations", [])
        if len(fixations) > 0:
            if self.curr_fixation is None:
                self.curr_fixation = fixations[0]
            else:
                if self.curr_fixation["id"] == fixations[0]["id"]:
                    return
                else:
                    self.curr_fixation = fixations[0]
                    
        frame_data = events.get("frame.world", [])
        if len(fixations) > 0 and len(frame_data) > 0:
            raw_data = frame_data[-1].get("__raw_data__")
            if raw_data:
                raw_image_bytes = bytes(raw_data[0])
                image = cv2.imdecode(np.frombuffer(raw_image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                if image is not None:
                    results = self.model(image)
                    fix_norm_pos = fixations[0]['norm_pos']
                    obj_gaze = get_object_in_gaze(fix_norm_pos, results)
                    
                    custom_datum = {
                        "topic": CUSTOM_TOPIC,
                        "timestamp": self.g_pool.get_timestamp(),  # Timestamp in pupil time
                        "object": obj_gaze,
                        "fixation": fix_norm_pos
                    }
                    events[CUSTOM_TOPIC] = [custom_datum]