import logging
import cv2
import numpy as np
from pyglui import ui
from pyglui.pyfontstash import fontstash
import OpenGL.GL as gl
from ultralytics import YOLO
from plugin import Plugin
from pyglui.cygl.utils import draw_polyline_norm, RGBA
from PIL import Image


class GazeObjectDetector(Plugin):
    CUSTOM_TOPIC = "gaze_object"

    icon_chr = chr(0xEC07)
    icon_font = "pupil_icons"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.index = 0
        self.max_duration = 0.4
        self.min_prob = 0.5
        self.vis_object_bbox = False
        self.glfont = None

        self.gaze_buffer = []
        self.fixation_buffer = []
        self.mean_fixation = None
        self.curr_fixation = None
        self.curr_object = None
        self.last_object = None
        self.curr_bbox = None
        self.curr_results = None

        self.model_path = "yolo11s.pt"
        self.model = YOLO(self.model_path)
        self.logger.info("GazeObjectDetector initialized.")

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Gaze Object Detector"

    def reinit_model(self):
        self.model = YOLO(self.model_path)
        self.logger.info(f"Model reinitialized with path: {self.model_path}")

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Gaze Object Detection Parameters"
        self.menu.append(
            ui.Info_Text("This plugin detects objects in the fixation of the user.")
        )
        self.menu.append(
            ui.Switch("vis_object_bbox", self, label="Visualize Bounding Box")
        )
        self.menu.append(
            ui.Info_Text("For the object detection we rely on the YOLO compatible models. The default model is "
                         "YOLOv11n, you can specify your own model path.")
        )
        self.menu.append(
            ui.Text_Input("model_path", self, label="Model Path")
        )
        self.menu.append(
            ui.Slider("min_prob", self, min=0.00, step=0.05, max=0.9, label="Min Confidence")
        )

        self.menu.append(
            ui.Button("Reinitialize the Yolo model", self.reinit_model)
        )
        self.menu.append(
            ui.Info_Text("You can set the max duration of a fixation (has to be same or higher as the duration of the "
                         "fixation detector).")
        )
        self.menu.append(
            ui.Slider("max_duration", self, min=0.01, step=0.01, max=2.0, label="Max Duration")
        )
        # Initialize font context for text rendering
        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_size(30)
        self.glfont.set_color_float((0.8, 0.2, 0.2, 1.0))

    def gl_display(self):
        # Example coordinates (replace with dynamic coordinates as needed)
        if self.curr_object is not None and self.curr_fixation is not None:
            text = f"{self.curr_fixation['id']} - {self.curr_object.upper()}"
            norm_pos_1 = 20, 100
            norm_pos_2 = (20 + len(text) * 10), 100
            draw_polyline_norm(
                np.array([norm_pos_1, norm_pos_2]),
                thickness=30.0, color=RGBA(0.8, 0.2, 0.2, 1.0), line_type=gl.GL_LINES
            )
            self.glfont.draw_text(20, 100, text)

            if self.vis_object_bbox and self.curr_bbox is not None:
                x_min, y_min, x_max, y_max = self.curr_bbox

                y_min = 1 - y_min
                y_max = 1 - y_max

                # Define the bounding box as a list of points (polygon)
                bbox_points = np.array([
                    [x_min, y_min],  # Bottom-left
                    [x_max, y_min],  # Bottom-right
                    [x_max, y_min],  # Bottom-right
                    [x_max, y_max],  # Top-right
                    [x_max, y_max],  # Top-right
                    [x_min, y_max],  # Top-left
                    [x_min, y_max],  # Top-left
                    [x_min, y_min]  # Closing the loop (back to bottom-left)
                ])
                draw_polyline_norm(
                    bbox_points,
                    thickness=3.0, color=RGBA(0.8, 0.2, 0.2, 1.0), line_type=gl.GL_LINES
                )

    def deinit_ui(self):
        self.remove_menu()
        self.glfont = None

    def _predict_frame(self, raw):
        raw_image_bytes = bytes(raw)
        frame = cv2.imdecode(np.frombuffer(raw_image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        return self.model(frame, verbose=False)

    def _get_object_in_gaze(self, gaze):
        for r in self.curr_results:
            boxes = r.boxes
            classes = boxes.cls
            confs = boxes.conf
            obj_name = None
            curr_conf = 0
            for i, box in enumerate(boxes.xyxyn):
                # Check if the gaze point lies within the bounding box of detected objects
                if box[0] <= gaze[0] <= box[2] and box[1] <= (1 - gaze[1]) <= box[3]:
                    if confs[i] > self.min_prob and confs[i] > curr_conf:
                        obj_name = r.names[int(classes[i])]
                        self.curr_bbox = np.array(box.cpu())
                        curr_conf = confs[i]
            if obj_name is not None:
                return obj_name
        return None

    def _publish_event(self, events):
        custom_datum = {
            "topic": self.CUSTOM_TOPIC,
            "timestamp": self.g_pool.get_timestamp(),
            "id": self.index,
            "object": self.curr_object,
            "bbox": [float(x) for x in self.curr_bbox],
            "norm_pos": self.mean_fixation
        }
        events[self.CUSTOM_TOPIC] = [custom_datum]

    def recent_events(self, events):
        fixations = events.get("fixations", [])
        # Case 1: Fixation detected and current fixation is None
        if len(fixations) > 0 and self.curr_fixation is None:
            self.curr_fixation = fixations[0]
            self.index = self.curr_fixation["id"]
            self.fixation_buffer.append(self.curr_fixation)
            self.logger.debug(f"Received new fixation id: {self.curr_fixation['id']}")
            return None
        # Case 2: Fixation detected and current fixation is not None
        elif len(fixations) > 0 and self.curr_fixation is not None:
            if self.curr_fixation["id"] == fixations[0]["id"]:
                self.curr_fixation = fixations[0]
                self.fixation_buffer.append(fixations[0])
                if len(self.fixation_buffer) > 10:
                    if self.curr_fixation['timestamp'] - self.fixation_buffer[0]['timestamp'] > self.max_duration:
                        self.logger.debug(f"Fixation duration exceeded 1 second.")
                    else:
                        return None
                else:
                    return None
            else:
                self.logger.debug(f"Received new fixation id: {fixations[0]['id']}")

        # Case 3: No fixation detected
        elif len(fixations) == 0 and self.curr_fixation is None:
            return None

        # Compute the mean fixation position
        self.mean_fixation = list(np.mean([f["norm_pos"] for f in self.fixation_buffer], axis=0))

        frame_data = events.get("frame.world", [])
        if len(frame_data) > 0:
            raw_data = frame_data[-1].get("__raw_data__")
            self.curr_results = self._predict_frame(raw_data[0])
            self.curr_object = self._get_object_in_gaze(self.mean_fixation)
            if self.curr_object is not None:
                self.logger.debug(f"Fixation {self.curr_fixation['id']} on {self.curr_object}")
                self._publish_event(events)
            else:
                self.logger.debug("No object detected in gaze.")
            self.fixation_buffer = []
            self.curr_fixation = None
        else:
            self.logger.debug("No image data received.")

