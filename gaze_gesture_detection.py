import logging
from plugin import Plugin
from methods import denormalize
import cv2
import numpy as np
from pyglui import ui
from pyglui.cygl.utils import draw_circle, draw_points, RGBA, draw_rounded_rect
from pyglui.pyfontstash import fontstash


def cross_product(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def is_point_in_triangle(p1, p2, p3, p):
    def sign(a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])

    b1 = sign(p, p1, p2) < 0.0
    b2 = sign(p, p2, p3) < 0.0
    b3 = sign(p, p3, p1) < 0.0

    return (b1 == b2) and (b2 == b3)


class GazeGestureDetector(Plugin):
    CUSTOM_TOPIC = "gaze_gesture"
    TRIGGER_TOPIC = "gaze_object"

    icon_chr = chr(0xEC11)
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

        self.origin_distance_tr = 0.15
        self.gesture_duration = 0.7
        self.gap_max = 0.04
        self.gap_counter = 0
        self.error_tol = 20

        self.viz_text = ""

        self.gaze_buffer = []
        self.trigger_event = None
        self.curr_time = None
        self.curr_object = None
        self.direction = None
        self.origin_pos = None
        self.l_1 = None
        self.l_2 = None
        self.r_1 = None
        self.r_2 = None

        self.logger.info("GazeGestureDetector initialized.")

    @classmethod
    def parse_pretty_class_name(cls) -> str:
        return "Gaze Gesture Detector"

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Define Head Gesture Parameters"
        self.menu.append(
            ui.Info_Text("Head gestures are head movements while fixation on a object. A gesture is completed when the user "
                         "moves his head away and then back to the defined origin around the fixation.")
        )
        self.menu.append(
            ui.Info_Text("We can set a radius around the origin for the gaze points to be considered in the origin.")
        )
        self.menu.append(
            ui.Slider("origin_distance_tr", self, min=0.01, step=0.01, max=1, label="Origin Radius")
        )
        self.menu.append(
            ui.Info_Text("We can set the duration of the gesture which starts when the gaze moves out of the origin and "
                         "ends when he gaze returns back to the origin.")
        )
        self.menu.append(
            ui.Slider("gesture_duration", self, min=0.1, step=0.1, max=1, label="Gesture Max Duration")
        )
        self.menu.append(
            ui.Info_Text("The gap threshold is the maximum allowed distance between two gaze points. "
                         "We only want to consider smooth persecute movements.")
        )
        self.menu.append(
            ui.Slider("gap_max", self, min=0.01, step=0.01, max=0.2, label="Max Gap Threshold")
        )
        self.menu.append(
            ui.Slider("error_tol", self, min=1, step=1, max=100, label="Error Tolerance")
        )

        # Initialize font context for text rendering
        self.glfont = fontstash.Context()
        self.glfont.add_font("opensans", ui.get_opensans_font_path())
        self.glfont.set_size(30)
        self.glfont.set_color_float((0.2, 0.8, 0.2, 1.0))  # Green text

    def gl_display(self):
        self.glfont.draw_text(20, 130, self.viz_text)

    def deinit_ui(self):
        self.remove_menu()
        self.glfont = None

    def _gaze_in_origin(self, gaze):
        origin_distance = np.linalg.norm(np.array(gaze['norm_pos']) - np.array(self.origin_pos))
        if origin_distance < self.origin_distance_tr:
            return True
        else:
            return False

    def _check_distance_threshold(self, gaze):
        if len(self.gaze_buffer) == 0:
            distance = np.linalg.norm(np.array(gaze['norm_pos']) - np.array(self.origin_pos))
            if distance < (self.origin_distance_tr + self.gap_max):
                return True
            else:
                return False
        elif len(self.gaze_buffer) > 0:
            last_gaze = self.gaze_buffer[-1]
            distance = np.linalg.norm(np.array(gaze['norm_pos']) - np.array(last_gaze['norm_pos']))
            if distance < self.gap_max:
                return True
            else:
                return False
        return False

    def _detect_direction(self, point):
        if is_point_in_triangle(self.l_2, self.l_1, self.origin_pos, point):
            return "right"
        elif is_point_in_triangle(self.r_2, self.r_1, self.origin_pos, point):
            return "left"
        elif is_point_in_triangle(self.l_1, self.r_1, self.origin_pos, point):
            return "down"
        elif is_point_in_triangle(self.l_2, self.r_2, self.origin_pos, point):
            return "up"
        else:
            return self.direction

    def _publish_event(self, events):
        custom_datum = {
            "topic": self.CUSTOM_TOPIC,
            "timestamp": self.g_pool.get_timestamp(),
            "object": self.curr_object,
            "direction": self.direction
        }
        events[self.CUSTOM_TOPIC] = [custom_datum]

    def set_viz_text(self):
        if None not in [self.trigger_event, self.direction, self.curr_object]:
            self.viz_text = (f"{self.trigger_event['id']} -  {self.direction.upper()} on "
                             f"{self.curr_object.upper()}")

    def recent_events(self, events):
        if len(events['gaze']) == 0:
            return
        if self.TRIGGER_TOPIC not in events and self.trigger_event is None:
            return

        if self.TRIGGER_TOPIC in events and self.trigger_event is None:
            self.logger.debug("Starting Gesture Recognition.")
            self.trigger_event = events[self.TRIGGER_TOPIC][-1]
            self.curr_object = self.trigger_event['object']
            self.curr_time = self.g_pool.get_timestamp()
            self.origin_pos = self.trigger_event['norm_pos']
            self.l_1 = [self.origin_pos[0] - 2, self.origin_pos[1] + 2]
            self.l_2 = [self.origin_pos[0] - 2, self.origin_pos[1] - 2]
            self.r_1 = [self.origin_pos[0] + 2, self.origin_pos[1] + 2]
            self.r_2 = [self.origin_pos[0] + 2, self.origin_pos[1] - 2]

        if self.trigger_event is not None:
            curr_time = self.g_pool.get_timestamp()
            for gaze in events['gaze']:
                if self._gaze_in_origin(gaze) and len(self.gaze_buffer) == 0:
                    self.curr_time = self.g_pool.get_timestamp()
                    return
                elif not self._gaze_in_origin(gaze):
                    if self._check_distance_threshold(gaze):
                        self.gaze_buffer.append(gaze)
                        self.gap_counter = 0
                    else:
                        self.gap_counter += 1
                        if self.gap_counter > self.error_tol:

                            self.trigger_event = None
                            self.gaze_buffer = []

                            self.logger.debug(f"STOP: max_gap  exceeded with error_tol {self.error_tol}.")
                            return
                elif self._gaze_in_origin(gaze) and len(self.gaze_buffer) > 0:
                    if curr_time - self.curr_time < self.gesture_duration:
                        mean_gaze = np.mean([g['norm_pos'] for g in self.gaze_buffer], axis=0)
                        self.direction = self._detect_direction(mean_gaze)

                        self.set_viz_text()
                        self._publish_event(events)
                        self.logger.debug(f"Gesture on {self.curr_object.upper()} detected: {self.direction.upper()} head movement")
                    else:
                        self.logger.debug(f"STOP: The gesture was longer than the duration threshold.")
                    self.trigger_event = None
                    self.gaze_buffer = []
