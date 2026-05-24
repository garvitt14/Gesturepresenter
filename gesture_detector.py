import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

SWIPE_THRESHOLD  = 0.10   # fraction of frame width (normalised coords)
SWIPE_WINDOW     = 0.6    # seconds to detect a swipe
PINCH_THRESHOLD  = 0.07   # normalised distance between thumb and index
GESTURE_COOLDOWN = 0.8    # seconds before same gesture fires again


class GestureDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.position_history = []       # (norm_x, timestamp)
        self.last_gesture_time = {}
        self.laser_pos = None

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture   = "none"
        laser_pos = None

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # Raw normalised landmarks (0-1)
            pts_norm = [(p.x, p.y) for p in lm.landmark]
            # Pixel landmarks for drawing
            pts_px   = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]

            fingers = self._fingers_up(pts_norm)
            wrist_x = pts_norm[0][0]
            self._update_history(wrist_x)

            raw = self._classify(fingers, pts_norm)
            gesture = self._apply_cooldown(raw)

            if raw == "laser":
                laser_pos = pts_norm[8]   # index fingertip

            self._draw_label(frame, gesture, fingers)
        else:
            self.position_history.clear()

        return frame, gesture, laser_pos

    # ── Finger state ─────────────────────────────────────────────────── #

    def _fingers_up(self, pts):
        """
        Returns [thumb, index, middle, ring, pinky] as bools.
        Uses normalised coords so it works at any resolution.
        All four fingers: tip Y < pip Y means finger is raised
        (smaller Y = higher on screen).
        Thumb: compare tip X vs MCP X for a horizontal open check.
        """
        # Index=8/6, Middle=12/10, Ring=16/14, Pinky=20/18
        index  = pts[8][1]  < pts[6][1]
        middle = pts[12][1] < pts[10][1]
        ring   = pts[16][1] < pts[14][1]
        pinky  = pts[20][1] < pts[18][1]

        # Thumb: tip further from palm centre than IP joint
        # Use distance from wrist to decide
        thumb_tip_dist = self._dist(pts[4],  pts[0])
        thumb_ip_dist  = self._dist(pts[3],  pts[0])
        thumb = thumb_tip_dist > thumb_ip_dist * 1.1

        return [thumb, index, middle, ring, pinky]

    # ── Classification ───────────────────────────────────────────────── #

    def _classify(self, fingers, pts):
        thumb, index, middle, ring, pinky = fingers
        count = sum(fingers)

        # ── Swipe FIRST — highest priority ──
        # Only trigger swipe if hand is actually moving fast
        swipe = self._detect_swipe()
        if swipe:
            return swipe

        # ── Pinch ──
        pinch_dist = self._dist(pts[4], pts[8])
        if pinch_dist < PINCH_THRESHOLD and not middle:
            return "pinch_in"

        # ── Fist ──
        if count == 0:
            return "blank"

        # ── Open palm ──
        if count >= 4:
            return "pause"

        # ── Laser ──
        if index and not middle and not ring and not pinky:
            return "laser"

        return "none"
    # ── Swipe ────────────────────────────────────────────────────────── #

    def _update_history(self, x):
        now = time.time()
        self.position_history.append((x, now))
        self.position_history = [
            (px, pt) for px, pt in self.position_history
            if now - pt < SWIPE_WINDOW
        ]

    def _detect_swipe(self):
        if len(self.position_history) < 4:
            return None
        oldest = self.position_history[0][0]
        newest = self.position_history[-1][0]
        delta  = newest - oldest          # normalised, so 0.15 = 15% of frame
        if delta >  SWIPE_THRESHOLD: return "next"
        if delta < -SWIPE_THRESHOLD: return "prev"
        return None

    # ── Cooldown ─────────────────────────────────────────────────────── #

    def _apply_cooldown(self, gesture):
        if gesture in ("none", "laser", "pinch_in","pause", "blank"):
            return gesture
        now  = time.time()
        last = self.last_gesture_time.get(gesture, 0)
        if now - last < GESTURE_COOLDOWN:
            return "none"
        self.last_gesture_time[gesture] = now
        return gesture

    # ── Helpers ──────────────────────────────────────────────────────── #

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

        # Gesture label
        cv2.rectangle(frame, (10, 10), (220, 50), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (220, 50), color, 2)
        cv2.putText(frame, label, (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Finger state bar — shows which fingers are detected as up
        labels = ["T", "I", "M", "R", "P"]
        for i, (up, name) in enumerate(zip(fingers, labels)):
            x = 10 + i * 36
            c = (0, 220, 0) if up else (60, 60, 60)
            cv2.rectangle(frame, (x, 58), (x + 28, 90), c, -1)
            cv2.putText(frame, name, (x + 7, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# ── Run directly to test ──────────────────────────────────────────────── #
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = GestureDetector()

    print("Running — press Q to quit")
    print("Gestures: point(laser) | open palm(pause) | fist(blank) | swipe L/R | pinch")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        frame, gesture, laser = detector.process_frame(frame)
        if gesture != "none":
            print(f"  → {gesture}")
        cv2.imshow("Gesture Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()