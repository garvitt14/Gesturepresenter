import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

SWIPE_THRESHOLD      = 0.12
SWIPE_WINDOW         = 0.5
PINCH_THRESHOLD      = 0.07
GESTURE_COOLDOWN     = 1.0
HOLD_FRAMES_REQUIRED = 10


class GestureDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.position_history     = []
        self.last_gesture_time    = {}
        self.gesture_hold_counter = 0
        self.last_seen_gesture    = "none"

    def process_frame(self, frame):
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture   = "none"
        laser_pos = None

        if results.multi_hand_landmarks:
            lm       = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            pts_norm = [(p.x, p.y) for p in lm.landmark]
            fingers  = self._fingers_up(pts_norm)
            self._update_history(pts_norm[0][0])   # track wrist X

            raw = self._classify(fingers, pts_norm)

            # Laser and swipes are instant — no hold filter, no cooldown
            if raw == "laser":
                gesture   = "laser"
                laser_pos = pts_norm[8]             # index fingertip

            elif raw in ("next", "prev"):
                now  = time.time()
                last = self.last_gesture_time.get(raw, 0)
                if now - last >= GESTURE_COOLDOWN:
                    self.last_gesture_time[raw] = now
                    gesture = raw
                    self.gesture_hold_counter = 0
                    self.last_seen_gesture    = "none"

            else:
                # pause / blank / pinch_in — require hold
                if raw == self.last_seen_gesture:
                    self.gesture_hold_counter += 1
                else:
                    self.gesture_hold_counter = 1
                    self.last_seen_gesture    = raw

                if self.gesture_hold_counter >= HOLD_FRAMES_REQUIRED:
                    now  = time.time()
                    last = self.last_gesture_time.get(raw, 0)
                    cooldowns = {"pause": 2.0, "blank": 2.0, "pinch_in": 2.5}
                    cd = cooldowns.get(raw, GESTURE_COOLDOWN)
                    if now - last >= cd:
                        self.last_gesture_time[raw] = now
                        gesture = raw

            self._draw_label(frame, gesture, fingers)

        else:
            self.position_history.clear()
            self.gesture_hold_counter = 0
            self.last_seen_gesture    = "none"

        return frame, gesture, laser_pos

    # ── Finger state ─────────────────────────────────────────────────── #

    def _fingers_up(self, pts):
        index  = pts[8][1]  < pts[6][1]
        middle = pts[12][1] < pts[10][1]
        ring   = pts[16][1] < pts[14][1]
        pinky  = pts[20][1] < pts[18][1]
        thumb  = self._dist(pts[4], pts[0]) > self._dist(pts[3], pts[0]) * 1.1
        return [thumb, index, middle, ring, pinky]

    # ── Classification ───────────────────────────────────────────────── #

    def _classify(self, fingers, pts):
        thumb, index, middle, ring, pinky = fingers
        count = sum(fingers)

        # Swipe — velocity based, highest priority
        velocity = self._get_velocity()
        if abs(velocity) > 0.25:
            swipe = self._detect_swipe()
            if swipe:
                return swipe

        # Fist — before pinch
        if count == 0:
            return "blank"

        # Pinch
        if self._dist(pts[4], pts[8]) < PINCH_THRESHOLD and not middle:
            return "pinch_in"

        # Open palm
        if count >= 4:
            return "pause"

        # Laser — only index up
        if index and not middle and not ring and not pinky:
            return "laser"

        return "none"

    # ── Swipe helpers ────────────────────────────────────────────────── #

    def _update_history(self, x):
        now = time.time()
        self.position_history.append((x, now))
        self.position_history = [
            (px, pt) for px, pt in self.position_history
            if now - pt < SWIPE_WINDOW
        ]

    def _get_velocity(self):
        if len(self.position_history) < 2:
            return 0
        dt = self.position_history[-1][1] - self.position_history[0][1]
        dx = self.position_history[-1][0] - self.position_history[0][0]
        return 0 if dt < 0.01 else dx / dt

    def _detect_swipe(self):
        if len(self.position_history) < 4:
            return None
        delta = self.position_history[-1][0] - self.position_history[0][0]
        if delta > SWIPE_THRESHOLD:
            self.position_history.clear()
            return "next"
        if delta < -SWIPE_THRESHOLD:
            self.position_history.clear()
            return "prev"
        return None

    # ── Drawing ──────────────────────────────────────────────────────── #

    @staticmethod
    def _dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def _draw_label(self, frame, gesture, fingers):
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
            # Move system cursor — works when this window is NOT focused
            mx = int(laser[0] * sw)
            my = int(laser[1] * sh)
            pyautogui.moveTo(mx, my, duration=0)
            # Also draw red dot on frame for visual feedback
            fx = int(laser[0] * 640)
            fy = int(laser[1] * 480)
            cv2.circle(frame, (fx, fy), 12, (0, 0, 255), -1)

        elif gesture != "none":
            print(f"  → {gesture}")

        cv2.imshow("Gesture Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()