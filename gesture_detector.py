import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

GESTURE_COOLDOWN     = 1.0
HOLD_FRAMES_REQUIRED = 6.0
SWIPE_MIN_DIST       = 0.15
SWIPE_MAX_TIME       = 0.6


class GestureDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.last_gesture_time    = {}
        self.gesture_hold_counter = 0
        self.last_seen_gesture    = "none"
        self.swipe_start_x        = None
        self.swipe_start_time     = None

    def process_frame(self, frame):
        h, w    = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture   = "none"
        laser_pos = None

        if results.multi_hand_landmarks:
            lm      = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            pts     = [(p.x, p.y) for p in lm.landmark]
            fingers = self._fingers_up(pts)
            wx      = pts[0][0]
            swipe   = self._check_swipe(wx)

            raw = self._classify(fingers, pts, swipe)

            if raw == "laser":
                gesture   = "laser"
                laser_pos = pts[8]

            elif raw in ("next", "prev"):
                now  = time.time()
                last = self.last_gesture_time.get(raw, 0)
                if now - last >= GESTURE_COOLDOWN:
                    self.last_gesture_time[raw] = now
                    gesture = raw
                    self.gesture_hold_counter = 0
                    self.last_seen_gesture    = "none"
                    self.swipe_start_x        = None
                    self.swipe_start_time     = None

            else:
                if raw == self.last_seen_gesture:
                    self.gesture_hold_counter += 1
                else:
                    self.gesture_hold_counter = 1
                    self.last_seen_gesture    = raw

                hold_frames = {"pinch_in": 6}
                cooldowns   = {"pause": 2.0, "blank": 2.0, "pinch_in": 2.5}

                if self.gesture_hold_counter >= hold_frames.get(raw, HOLD_FRAMES_REQUIRED):
                    cd   = cooldowns.get(raw, GESTURE_COOLDOWN)
                    now  = time.time()
                    last = self.last_gesture_time.get(raw, 0)
                    if now - last >= cd:
                        self.last_gesture_time[raw] = now
                        gesture = raw

            self._draw_label(frame, gesture, fingers)

        else:
            self.gesture_hold_counter = 0
            self.last_seen_gesture    = "none"
            self.swipe_start_x        = None
            self.swipe_start_time     = None

        return frame, gesture, laser_pos

    # ── Swipe ────────────────────────────────────────────────────────── #

    def _check_swipe(self, wx):
        now = time.time()
        if self.swipe_start_x is None:
            self.swipe_start_x    = wx
            self.swipe_start_time = now
            return None
        elapsed = now - self.swipe_start_time
        dist    = wx - self.swipe_start_x
        if elapsed > SWIPE_MAX_TIME:
            self.swipe_start_x    = wx
            self.swipe_start_time = now
            return None
        if dist > SWIPE_MIN_DIST:
            self.swipe_start_x    = wx
            self.swipe_start_time = now
            return "next"
        if dist < -SWIPE_MIN_DIST:
            self.swipe_start_x    = wx
            self.swipe_start_time = now
            return "prev"
        return None

    # ── Finger state ─────────────────────────────────────────────────── #

    def _fingers_up(self, pts):
        index  = pts[8][1]  < pts[6][1]
        middle = pts[12][1] < pts[10][1]
        ring   = pts[16][1] < pts[14][1]
        pinky  = pts[20][1] < pts[18][1]
        thumb  = self._dist(pts[4], pts[0]) > self._dist(pts[3], pts[0]) * 1.1
        return [thumb, index, middle, ring, pinky]

    # ── Classification ───────────────────────────────────────────────── #

    def _classify(self, fingers, pts, swipe):
        thumb, index, middle, ring, pinky = fingers
        count = sum(fingers)

        pinch_dist = self._dist(pts[4], pts[8])

        # ── Pinch FIRST — before fist, because a pinch curls fingers
        #    and can look like count==0. Thumb must be somewhat raised
        #    (not fully tucked) and tip-to-tip distance under threshold.
        thumb_dist_from_palm = self._dist(pts[4], pts[0])
        thumb_raised = thumb_dist_from_palm > self._dist(pts[2], pts[0]) * 1.05
        if pinch_dist < 0.11 and thumb_raised:
            return "pinch_in"

        # Two fingers = swipe gesture
        is_swipe_shape = index and middle and not ring and not pinky
        if is_swipe_shape and swipe:
            return swipe

        # Fist — thumb must also be down to avoid false positives
        if count == 0 and not thumb_raised:
            return "blank"

        # Open palm
        if count >= 4:
            return "pause"

        # Laser
        if index and not middle and not ring and not pinky:
            return "laser"

        if is_swipe_shape:
            return "none"

        return "none"

    # ── Helpers ──────────────────────────────────────────────────────── #

    @staticmethod
    def _dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def _draw_label(self, frame, gesture, fingers):
        if not gesture:
            gesture = "none"
        colors = {
            "next":     (0,   255, 100),
            "prev":     (0,   180, 255),
            "laser":    (0,   0,   255),
            "pause":    (255, 200, 0  ),
            "blank":    (180, 180, 180),
            "pinch_in": (255, 100, 255),
            "none":     (80,  80,  80 ),
        }
        color = colors.get(gesture, (80, 80, 80))
        label = gesture.upper().replace("_", " ")
        cv2.rectangle(frame, (10, 10), (220, 50), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (220, 50), color, 2)
        cv2.putText(frame, label, (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        names = ["T", "I", "M", "R", "P"]
        for i, (up, name) in enumerate(zip(fingers, names)):
            x = 10 + i * 36
            c = (0, 220, 0) if up else (60, 60, 60)
            cv2.rectangle(frame, (x, 58), (x + 28, 90), c, -1)
            cv2.putText(frame, name, (x + 7, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# ── Test ─────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    import pyautogui
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = GestureDetector()
    sw, sh   = pyautogui.size()
    print("Running — press Q to quit")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        frame, gesture, laser = detector.process_frame(frame)

        if gesture == "laser" and laser is not None:
            mx = int(laser[0] * sw)
            my = int(laser[1] * sh)
            pyautogui.moveTo(mx, my, duration=0)
            fx = int(laser[0] * 640)
            fy = int(laser[1] * 480)
            cv2.circle(frame, (fx, fy), 12, (0, 0, 255), -1)
        elif gesture != "none":
            print(f"  >>> GESTURE: {gesture}")

        cv2.imshow("Gesture Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()