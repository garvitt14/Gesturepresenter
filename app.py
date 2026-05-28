"""
GesturePresenter — Web App
Run:  python app.py
Open: http://localhost:5000
"""

import cv2
import threading
import time
import os
from flask import Flask, Response, render_template_string, jsonify

from gesture_detector  import GestureDetector
from slide_controller  import SlideController

app        = Flask(__name__)
detector   = GestureDetector()
controller = SlideController()

# ── Shared state ─────────────────────────────────────────────────────── #
state = {
    "gesture":     "none",
    "last_action": "",
    "action_time": 0,
}
state_lock  = threading.Lock()
frame_lock  = threading.Lock()
current_frame = None

# ── Camera loop ──────────────────────────────────────────────────────── #
def camera_loop():
    global current_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("Camera opened.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        frame, gesture, laser_pos = detector.process_frame(frame)

        # Draw laser dot on frame
        if gesture == "laser" and laser_pos is not None:
            fx = int(laser_pos[0] * 640)
            fy = int(laser_pos[1] * 480)
            cv2.circle(frame, (fx, fy), 14, (0, 0, 255), -1)
            cv2.circle(frame, (fx, fy), 18, (255, 255, 255), 2)

        action = controller.handle(gesture, laser_pos)

        with state_lock:
            state["gesture"] = gesture
            if action:
                state["last_action"] = action
                state["action_time"] = time.time()

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            current_frame = buf.tobytes()

threading.Thread(target=camera_loop, daemon=True).start()

# ── Stream ───────────────────────────────────────────────────────────── #
def gen_frames():
    while True:
        with frame_lock:
            frame = current_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/state")
def get_state():
    with state_lock:
        s = dict(state)
    # Clear action after 2 seconds so it doesn't show forever
    if time.time() - s["action_time"] > 2:
        s["last_action"] = ""
    return jsonify(s)

# ── HTML ─────────────────────────────────────────────────────────────── #
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GesturePresenter</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #080810;
    --surface:  #111118;
    --surface2: #1a1a24;
    --border:   #252535;
    --accent:   #6c63ff;
    --accent2:  #ff6584;
    --green:    #4ade80;
    --amber:    #fbbf24;
    --red:      #f87171;
    --blue:     #60a5fa;
    --pink:     #e879f9;
    --gray:     #94a3b8;
    --text:     #e2e8f0;
    --muted:    #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* Subtle dot grid */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image: radial-gradient(var(--border) 1px, transparent 1px);
    background-size: 28px 28px;
    opacity: 0.5;
    pointer-events: none;
    z-index: 0;
  }

  .wrap {
    position: relative; z-index: 1;
    max-width: 1280px; margin: 0 auto;
    padding: 20px;
    display: flex; flex-direction: column; gap: 16px;
  }

  /* Header */
  header {
    display: flex; align-items: center; gap: 14px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .logo {
    font-family: 'Space Mono', monospace;
    font-size: 20px; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .live-badge {
    font-family: 'Space Mono', monospace; font-size: 11px;
    background: rgba(74,222,128,0.15); color: var(--green);
    border: 1px solid rgba(74,222,128,0.3);
    padding: 3px 10px; border-radius: 20px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
  }
  .subtitle {
    margin-left: auto;
    font-size: 12px; color: var(--muted);
    font-family: 'Space Mono', monospace;
  }

  /* Main grid */
  .grid {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 16px;
    align-items: start;
  }
  @media (max-width: 860px) {
    .grid { grid-template-columns: 1fr; }
  }

  /* Camera */
  .camera-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    position: relative;
  }
  .camera-wrap img {
    width: 100%; display: block;
  }

  /* Gesture badge overlay on camera */
  .gesture-overlay {
    position: absolute; bottom: 16px; left: 16px;
    display: flex; align-items: center; gap: 10px;
  }
  .gesture-badge {
    font-family: 'Space Mono', monospace;
    font-size: 15px; font-weight: 700;
    padding: 8px 18px; border-radius: 10px;
    border: 2px solid currentColor;
    background: rgba(8,8,16,0.75);
    backdrop-filter: blur(8px);
    transition: color 0.2s, border-color 0.2s;
    min-width: 120px; text-align: center;
  }
  .action-badge {
    font-size: 13px; font-weight: 500;
    padding: 8px 14px; border-radius: 10px;
    background: rgba(8,8,16,0.75);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    color: var(--green);
    transition: opacity 0.3s;
    font-family: 'Space Mono', monospace;
  }

  /* Right panel */
  .panel { display: flex; flex-direction: column; gap: 14px; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
  }
  .card-title {
    font-family: 'Space Mono', monospace;
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 2px; color: var(--muted);
    margin-bottom: 14px;
  }

  /* Gesture guide */
  .gesture-list { display: flex; flex-direction: column; gap: 8px; }
  .gesture-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 12px; border-radius: 10px;
    background: var(--surface2);
    border: 1px solid transparent;
    transition: border-color 0.2s, background 0.2s;
  }
  .gesture-row.active {
    border-color: currentColor;
    background: rgba(255,255,255,0.04);
  }
  .gesture-emoji { font-size: 22px; width: 32px; text-align: center; }
  .gesture-info  { flex: 1; }
  .gesture-name  {
    font-size: 13px; font-weight: 600; margin-bottom: 2px;
  }
  .gesture-desc  { font-size: 11px; color: var(--muted); }
  .gesture-key   {
    font-family: 'Space Mono', monospace; font-size: 10px;
    background: var(--border); color: var(--muted);
    padding: 2px 7px; border-radius: 5px;
  }

  /* Status card */
  .status-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat {
    background: var(--surface2); border-radius: 10px;
    padding: 10px 12px;
  }
  .stat-label { font-size: 10px; color: var(--muted); margin-bottom: 4px; font-family:'Space Mono',monospace; }
  .stat-value { font-size: 15px; font-weight: 600; font-family:'Space Mono',monospace; }

  /* Tip card */
  .tip {
    font-size: 12px; color: var(--muted); line-height: 1.7;
  }
  .tip strong { color: var(--text); }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <span class="logo">GesturePresenter</span>
    <span class="live-badge">● LIVE</span>
    <span class="subtitle">Control slides with your hands</span>
  </header>

  <div class="grid">

    <!-- Camera feed -->
    <div class="camera-wrap">
      <img src="/video_feed" alt="Camera">
      <div class="gesture-overlay">
        <div class="gesture-badge" id="gesture-badge" style="color:#64748b">NONE</div>
        <div class="action-badge" id="action-badge" style="opacity:0">–</div>
      </div>
    </div>

    <!-- Right panel -->
    <div class="panel">

      <!-- Gesture guide -->
      <div class="card">
        <div class="card-title">Gesture Guide</div>
        <div class="gesture-list" id="gesture-list">

          <div class="gesture-row" id="row-next" style="color:var(--green)">
            <div class="gesture-emoji">👉</div>
            <div class="gesture-info">
              <div class="gesture-name">Swipe Right</div>
              <div class="gesture-desc">Next slide</div>
            </div>
            <div class="gesture-key">→</div>
          </div>

          <div class="gesture-row" id="row-prev" style="color:var(--blue)">
            <div class="gesture-emoji">👈</div>
            <div class="gesture-info">
              <div class="gesture-name">Swipe Left</div>
              <div class="gesture-desc">Previous slide</div>
            </div>
            <div class="gesture-key">←</div>
          </div>

          <div class="gesture-row" id="row-laser" style="color:var(--red)">
            <div class="gesture-emoji">☝️</div>
            <div class="gesture-info">
              <div class="gesture-name">Point</div>
              <div class="gesture-desc">Laser pointer</div>
            </div>
            <div class="gesture-key">mouse</div>
          </div>

          <div class="gesture-row" id="row-pause" style="color:var(--amber)">
            <div class="gesture-emoji">✋</div>
            <div class="gesture-info">
              <div class="gesture-name">Open Palm</div>
              <div class="gesture-desc">Blank screen toggle</div>
            </div>
            <div class="gesture-key">B</div>
          </div>

          <div class="gesture-row" id="row-blank" style="color:var(--gray)">
            <div class="gesture-emoji">✊</div>
            <div class="gesture-info">
              <div class="gesture-name">Fist</div>
              <div class="gesture-desc">White screen toggle</div>
            </div>
            <div class="gesture-key">W</div>
          </div>

          <div class="gesture-row" id="row-pinch" style="color:var(--pink)">
            <div class="gesture-emoji">🤏</div>
            <div class="gesture-info">
              <div class="gesture-name">Pinch</div>
              <div class="gesture-desc">Exit slideshow</div>
            </div>
            <div class="gesture-key">ESC</div>
          </div>

        </div>
      </div>

      <!-- Tips -->
      <div class="card">
        <div class="card-title">Tips</div>
        <div class="tip">
          <strong>Swipe</strong> — move your whole hand quickly across the frame<br>
          <strong>Hold</strong> — palm, fist, pinch need ~0.3s hold to trigger<br>
          <strong>Laser</strong> — point index finger, move to control cursor<br>
          <strong>Lighting</strong> — works best with good front lighting
        </div>
      </div>

    </div>
  </div>
</div>

<script>
const gestureColors = {
  next:     "var(--green)",
  prev:     "var(--blue)",
  laser:    "var(--red)",
  pause:    "var(--amber)",
  blank:    "var(--gray)",
  pinch_in: "var(--pink)",
  none:     "var(--muted)",
};

const gestureRows = {
  next:     "row-next",
  prev:     "row-prev",
  laser:    "row-laser",
  pause:    "row-pause",
  blank:    "row-blank",
  pinch_in: "row-pinch",
};

let lastGesture = "none";
let actionTimer = null;

async function poll() {
  try {
    const res  = await fetch("/state");
    const data = await res.json();

    const gesture = data.gesture || "none";
    const action  = data.last_action || "";

    // Update gesture badge
    const badge = document.getElementById("gesture-badge");
    badge.textContent  = gesture.toUpperCase().replace("_", " ");
    badge.style.color  = gestureColors[gesture] || "var(--muted)";
    badge.style.borderColor = gestureColors[gesture] || "var(--muted)";

    // Highlight active gesture row
    if (gesture !== lastGesture) {
      // Remove active from all
      document.querySelectorAll(".gesture-row").forEach(r => r.classList.remove("active"));
      // Add active to current
      const rowId = gestureRows[gesture];
      if (rowId) document.getElementById(rowId)?.classList.add("active");
      lastGesture = gesture;
    }

    // Show action badge
    const actionBadge = document.getElementById("action-badge");
    if (action) {
      actionBadge.textContent = action;
      actionBadge.style.opacity = "1";
      clearTimeout(actionTimer);
      actionTimer = setTimeout(() => {
        actionBadge.style.opacity = "0";
      }, 2000);
    }

  } catch(e) {}
}

setInterval(poll, 150);
poll();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    print("\n🎯 GesturePresenter running → http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
