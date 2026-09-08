#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest Group GmbH
# SPDX-License-Identifier: Apache-2.0
"""Web click-labeler: click the main action on each sampled frame.

    python eval/labeler/app.py --host 0.0.0.0 --port 8080

Reads the frame index produced by ``extract_frames.py`` and serves a single-page UI.
Every click (x, y, in full-frame pixels) is persisted to ``labels.json`` immediately, so
labeling is safe to stop and resume. Convert the finished labels into scorer-ready
``ground_truth.jsonl`` with ``ingest_labels.py``.

Requires Flask (dev dependency): ``uv sync`` installs it.
"""
import argparse
import json
import os
import threading

from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
# Working-file locations; overridable via CLI (must match extract_frames.py's outputs).
INDEX_PATH = os.path.join(HERE, "frames_index.json")
LABELS_PATH = os.path.join(HERE, "labels.json")
FRAMES_DIR = os.path.join(HERE, "static", "frames")

app = Flask(__name__)
lock = threading.Lock()


def load_frames():
    with open(INDEX_PATH) as f:
        return json.load(f)


def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            return json.load(f)
    return {}


def save_labels(labels):
    tmp = LABELS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(labels, f, indent=1)
    os.replace(tmp, LABELS_PATH)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Action labeler</title>
<style>
 body{font-family:system-ui;margin:0;background:#111;color:#eee;text-align:center}
 #bar{padding:10px;font-size:15px}
 #wrap{position:relative;display:inline-block;cursor:crosshair;max-width:100vw}
 #img{max-width:100vw;max-height:82vh;display:block}
 .dot{position:absolute;width:18px;height:18px;margin:-9px 0 0 -9px;border:3px solid #0f0;
      border-radius:50%;pointer-events:none}
 button{font-size:15px;margin:6px;padding:8px 18px;border-radius:8px;border:0;
        background:#333;color:#eee}
 #done{display:none;padding:40px;font-size:20px}
 .hint{color:#999;font-size:13px;max-width:640px;margin:4px auto}
</style></head><body>
<div id="bar"></div>
<div class="hint">Click the <b>main action</b>: the ball/puck if you can see it, otherwise the
ball-carrier / center of the active play. <b>Skip</b> if there is no clear action
(replay wipe, crowd, huddle). Click again to correct before pressing Next.</div>
<div id="wrap"><img id="img"><div id="dot" class="dot" style="display:none"></div></div>
<div>
 <button onclick="back()">&#8592; Back</button>
 <button onclick="skip()">Skip (no clear action)</button>
 <button id="next" onclick="next()" disabled>Next &#8594;</button>
</div>
<div id="done">All frames labeled — thank you! You can close this tab and run
ingest_labels.py.</div>
<script>
let frames=[], labels={}, cur=0, pending=null;
function key(f){return f.item_id+"__"+f.frame_idx}
async function init(){
  const s = await (await fetch('api/state')).json();
  frames = s.frames; labels = s.labels;
  cur = frames.findIndex(f => !(key(f) in labels));
  if (cur < 0) cur = frames.length;
  show();
}
function show(){
  if (cur >= frames.length){
    document.getElementById('wrap').style.display='none';
    document.getElementById('done').style.display='block';
    document.getElementById('bar').textContent='Done: '+frames.length+' / '+frames.length;
    return;
  }
  document.getElementById('wrap').style.display='inline-block';
  document.getElementById('done').style.display='none';
  const f = frames[cur];
  const n = Object.keys(labels).length;
  document.getElementById('bar').textContent =
    'Frame '+(cur+1)+' / '+frames.length+'  —  clip "'+f.item_id+'"  frame '+
    f.frame_idx+'   (labeled so far: '+n+')';
  const img = document.getElementById('img');
  img.src = 'frames/'+f.file;
  pending = labels[key(f)] || null;
  drawDot();
  document.getElementById('next').disabled = !pending;
}
function drawDot(){
  const dot = document.getElementById('dot'), img = document.getElementById('img');
  if (!pending || pending.skipped){ dot.style.display='none'; return; }
  const f = frames[cur];
  dot.style.display='block';
  dot.style.left = (pending.x / f.width * img.clientWidth) + 'px';
  dot.style.top  = (pending.y / f.height * img.clientHeight) + 'px';
}
document.getElementById('img').onload = drawDot;
document.getElementById('img').addEventListener('click', ev => {
  const img = ev.target, f = frames[cur];
  const r = img.getBoundingClientRect();
  pending = {x: (ev.clientX - r.left) / img.clientWidth  * f.width,
             y: (ev.clientY - r.top)  / img.clientHeight * f.height, skipped:false};
  drawDot();
  document.getElementById('next').disabled = false;
});
async function commit(label){
  const f = frames[cur];
  labels[key(f)] = label;
  await fetch('api/label', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({item_id:f.item_id, frame_idx:f.frame_idx, ...label})});
}
async function next(){ if(pending){ await commit(pending); cur++; show(); } }
async function skip(){ await commit({x:null,y:null,skipped:true}); cur++; show(); }
function back(){ if(cur>0){ cur--; show(); } }
window.addEventListener('resize', drawDot);
init();
</script></body></html>"""


@app.route("/")
def root():
    return PAGE


@app.route("/api/state")
def state():
    with lock:
        return jsonify({"frames": load_frames(), "labels": load_labels()})


@app.route("/api/label", methods=["POST"])
def label():
    d = request.get_json(force=True)
    frames = load_frames()
    if not d.get("skipped"):
        f = next((x for x in frames if x["item_id"] == d["item_id"]
                  and x["frame_idx"] == d["frame_idx"]), None)
        if f is None:
            return jsonify({"error": "unknown frame"}), 400
        if not (0 <= d["x"] <= f["width"] and 0 <= d["y"] <= f["height"]):
            return jsonify({"error": "click out of bounds"}), 400
    with lock:
        labels = load_labels()
        labels[f'{d["item_id"]}__{d["frame_idx"]}'] = {
            "x": d.get("x"), "y": d.get("y"), "skipped": bool(d.get("skipped"))}
        save_labels(labels)
    return jsonify({"ok": True, "n": len(labels)})


@app.route("/frames/<path:name>")
def frame_file(name):
    return send_from_directory(FRAMES_DIR, name)


def main(argv=None):
    global INDEX_PATH, LABELS_PATH, FRAMES_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--index", default=INDEX_PATH, help="frames_index.json from extract_frames.py")
    ap.add_argument("--labels", default=LABELS_PATH, help="labels.json to read/write (resume-safe)")
    ap.add_argument("--frames", default=FRAMES_DIR, help="dir holding the extracted JPEGs")
    args = ap.parse_args(argv)
    INDEX_PATH, LABELS_PATH, FRAMES_DIR = args.index, args.labels, args.frames
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
