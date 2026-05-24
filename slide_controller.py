import pyautogui
import time

pyautogui.PAUSE    = 0
pyautogui.FAILSAFE = True


class SlideController:
    def __init__(self):
        self.last_action_time = {}
        self.cooldowns = {
            "next":     0.8,
            "prev":     0.8,
            "pause":    2.0,   # longer — it's a toggle
            "blank":    2.0,
            "pinch_in": 2.5,   # longest — exits slideshow
        }
        self.blank_active        = False
        self.screen_w, self.screen_h = pyautogui.size()
        print(f"SlideController ready — screen: {self.screen_w}x{self.screen_h}")

    def handle(self, gesture: str, laser_pos=None):
        action_taken = ""

        if gesture == "next":
            if self._can_fire("next"):
                pyautogui.press("right")
                action_taken = "→ Next slide"

        elif gesture == "prev":
            if self._can_fire("prev"):
                pyautogui.press("left")
                action_taken = "← Prev slide"

        elif gesture == "pause":
            if self._can_fire("pause"):
                pyautogui.press("b")
                self.blank_active = not self.blank_active
                action_taken = "⬛ Blank" if self.blank_active else "▶ Resume"

        elif gesture == "blank":
            if self._can_fire("blank"):
                pyautogui.press("w")
                action_taken = "⬜ White screen"

        elif gesture == "pinch_in":
            if self._can_fire("pinch_in"):
                pyautogui.press("escape")
                action_taken = "✕ Exit slideshow"

        elif gesture == "laser" and laser_pos is not None:
            screen_x = int(laser_pos[0] * self.screen_w)
            screen_y = int(laser_pos[1] * self.screen_h)
            screen_x = max(0, min(screen_x, self.screen_w - 1))
            screen_y = max(0, min(screen_y, self.screen_h - 1))
            pyautogui.moveTo(screen_x, screen_y, duration=0)
            action_taken = f"🔴 Laser ({screen_x}, {screen_y})"

        return action_taken

    def _can_fire(self, action: str) -> bool:
        now      = time.time()
        last     = self.last_action_time.get(action, 0)
        cooldown = self.cooldowns.get(action, 0.8)
        if now - last >= cooldown:
            self.last_action_time[action] = now
            return True
        return False


if __name__ == "__main__":
    import cv2
    from gesture_detector import GestureDetector

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector   = GestureDetector()
    controller = SlideController()

    print("Open PowerPoint in slideshow mode, alt+tab back, then gesture.")
    print("Press Q to quit.")

    last_action = ""

    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        frame, gesture, laser_pos = detector.process_frame(frame)

        action = controller.handle(gesture, laser_pos)
        if action:
            last_action = action

        if last_action:
            cv2.rectangle(frame, (10, 440), (630, 475), (20, 20, 20), -1)
            cv2.putText(frame, f"Action: {last_action}", (16, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)

        cv2.imshow("Slide Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()