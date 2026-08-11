"""
PROJECT ANGELINA — Neural-Ergonomic Focus Bot
============================================================
SCRIPT 5: REAL-TIME WEB DASHBOARD
Purpose : Flask + SocketIO server that receives posture data
          from Script 3 and serves a live web dashboard.
============================================================
USAGE:
  Run this in a SEPARATE terminal alongside Script 3:
    Terminal 1: python 3_realtime_inference.py
    Terminal 2: python 5_dashboard.py

  Then open your browser to: http://localhost:5001
============================================================
INSTALL:
  pip install flask flask-socketio
============================================================
"""
import sys
if sys.stdout is not None and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time
import json
from datetime import datetime
from collections import deque
import os
import config

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins=f"http://localhost:{config.DASHBOARD_PORT}", async_mode='threading')

# ─────────────────────────────────────────────
#  SHARED STATE (updated by Script 3 via API)
# ─────────────────────────────────────────────
session_data = {
    'current_label':     'unknown',
    'current_confidence': 0.0,
    'session_start':      datetime.now().isoformat(),
    'total_triggers':     0,
    'posture_counts': {
        'good_posture':     0,
        'slouching':        0,
        'tech_neck':        0,
        'decaying_posture': 0,
        'unknown':          0,
    },
    'alert_log':    [],          # list of {time, label}
    'history':      deque(maxlen=120),  # last 120 data points for graph
}

lock = threading.Lock()


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/update', methods=['POST'])
def update():
    """
    Script 3 calls this endpoint every frame to push posture data.
    Add this to Script 3's main loop (see integration instructions below).
    """
    from flask import request
    data = request.get_json()

    with lock:
        label      = data.get('label', 'unknown')
        confidence = data.get('confidence', 0.0)
        triggered  = data.get('triggered', False)

        session_data['current_label']      = label
        session_data['current_confidence'] = confidence

        # Update posture counts
        if label in session_data['posture_counts']:
            session_data['posture_counts'][label] += 1

        # Log trigger events
        if triggered:
            session_data['total_triggers'] += 1
            session_data['alert_log'].append({
                'time':  datetime.now().strftime('%H:%M:%S'),
                'label': label,
            })
            # Keep only last 50 alerts
            if len(session_data['alert_log']) > 50:
                session_data['alert_log'] = session_data['alert_log'][-50:]

        # Add to history for graph
        session_data['history'].append({
            'time':       datetime.now().strftime('%H:%M:%S'),
            'label':      label,
            'confidence': round(confidence * 100, 1),
        })

        # Broadcast to all connected browsers via SocketIO (inside lock)
        emit_data = {
            'label':      label,
            'confidence': round(confidence * 100, 1),
            'triggered':  triggered,
            'counts':     dict(session_data['posture_counts']),
            'triggers':   session_data['total_triggers'],
            'alert_log':  session_data['alert_log'][-10:],
            'history':    list(session_data['history'])[-60:],
            'alert_frac': data.get('alert_frac', 0),
        }

    socketio.emit('posture_update', emit_data)

    return {'status': 'ok'}


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset session stats."""
    with lock:
        session_data['total_triggers'] = 0
        session_data['posture_counts'] = {
            'good_posture':     0,
            'slouching':        0,
            'tech_neck':        0,
            'decaying_posture': 0,
            'unknown':          0,
        }
        session_data['alert_log'] = []
        session_data['history']   = deque(maxlen=120)
        session_data['session_start'] = datetime.now().isoformat()
        new_start = session_data['session_start']
    socketio.emit('session_reset', {'session_start': new_start})
    return {'status': 'reset'}


@app.route('/api/state')
def state():
    """Return full current state (for page load)."""
    with lock:
        return {
            'label':      session_data['current_label'],
            'confidence': round(session_data['current_confidence'] * 100, 1),
            'counts':     session_data['posture_counts'],
            'triggers':   session_data['total_triggers'],
            'alert_log':  session_data['alert_log'][-10:],
            'history':    list(session_data['history'])[-60:],
            'session_start': session_data['session_start'],
        }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  PROJECT ANGELINA — Dashboard Server")
    print("="*55)
    print("  Open your browser to: http://localhost:5001")
    print("  Run Script 3 in another terminal to feed data.")
    print("="*55 + "\n")
    socketio.run(app, host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT, debug=False)


"""
============================================================
INTEGRATION: Add this to Script 3 (3_realtime_inference.py)
============================================================
Add this import at the top of Script 3:

    import requests

Add this constant near the top:

    DASHBOARD_URL = "http://localhost:5001/api/update"
    DASHBOARD_ENABLED = True

Then inside the main inference loop, after the label is determined,
add this block (after the draw_hud call):

    if DASHBOARD_ENABLED:
        try:
            requests.post(DASHBOARD_URL, json={
                'label':      label,
                'confidence': float(confidence),
                'triggered':  (bad_hold >= ALERT_HOLD_S) and (cooldown_remaining == 0.0),
            }, timeout=0.05)
        except Exception:
            pass   # dashboard offline — don't crash inference

"""