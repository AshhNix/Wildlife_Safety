from datetime import datetime
import io
import os
import re
import hashlib
import tempfile
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------------------
# WILDLIFE SAFETY: animated, app-style UI (pure CSS animation, no JS)
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Wildlife Safety | Protection & Risk Monitoring",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

FOREST_CLASSES = {"person"}
ROAD_ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe",
}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}

HAZARD_KEYWORDS = {
    "fire": ("fire", "flame", "wildfire", "inferno", "burning"),
    "smoke": ("smoke", "smog"),
    "volcanic": ("volcano", "eruption", "lava", "ash"),
    "pest_disease": ("pest", "bug", "insect", "locust", "disease", "infected", "fungus"),
}
ALL_TARGET_CLASSES = FOREST_CLASSES | ROAD_ANIMAL_CLASSES
HAZARD_CATEGORIES = {"Wildfire / Fire", "Smoke Detection", "Volcanic Hazard", "Pest / Disease"}


def classify_label(label: str) -> str:
    label = label.lower().strip()
    if label in FOREST_CLASSES:
        return "Human Intrusion"
    if label in ROAD_ANIMAL_CLASSES:
        return "Wildlife / Road Risk"
    for category, keywords in HAZARD_KEYWORDS.items():
        if any(re.search(r"(?<![a-z])" + re.escape(keyword), label) for keyword in keywords):
            return {
                "fire": "Wildfire / Fire",
                "smoke": "Smoke Detection",
                "volcanic": "Volcanic Hazard",
                "pest_disease": "Pest / Disease",
            }[category]
    return "Other Detection"


def severity_for(category: str, proximity: str) -> str:
    if category in {"Wildfire / Fire", "Smoke Detection", "Volcanic Hazard"}:
        return "CRITICAL"
    if category == "Human Intrusion":
        return "HIGH" if proximity in {"High", "Medium"} else "MODERATE"
    if category == "Wildlife / Road Risk":
        return "HIGH" if proximity == "High" else "MODERATE"
    if category == "Pest / Disease":
        return "MODERATE"
    return "LOW"


def color_for(category: str):
    # Annotated image is an RGB array, so these are RGB tuples.
    palette = {
        "Human Intrusion": (214, 52, 86),
        "Wildlife / Road Risk": (16, 150, 115),
        "Wildfire / Fire": (230, 40, 30),
        "Smoke Detection": (96, 104, 120),
        "Volcanic Hazard": (190, 60, 30),
        "Pest / Disease": (160, 120, 20),
        "Other Detection": (91, 100, 230),
    }
    return palette.get(category, (91, 100, 230))


def compute_proximity(box_area: float, frame_area: float) -> str:
    if frame_area <= 0:
        return "Unknown"
    ratio = box_area / frame_area
    if ratio >= 0.12:
        return "High"
    if ratio >= 0.03:
        return "Medium"
    return "Low"


def get_advisory(category: str, label: str) -> str:
    if category == "Human Intrusion":
        return "Possible unauthorized human presence. Verify the alert, check camera context, and notify the forest patrol if the zone is restricted."
    if category == "Wildlife / Road Risk":
        return f"{label.title()} detected near a monitored area. Reduce speed, keep distance, avoid sudden honking, and leave a safe escape route."
    if category == "Wildfire / Fire":
        return "Potential fire detected. Confirm with a second camera or field team and alert the control room immediately. Follow local emergency protocol."
    if category == "Smoke Detection":
        return "Possible smoke plume detected. Check for fire or other source and alert the control team for verification."
    if category == "Volcanic Hazard":
        return "Possible volcanic activity or related hazard detected. Verify the source and follow the area's official emergency and evacuation guidance."
    if category == "Pest / Disease":
        return "Possible pest or vegetation-disease indicator detected. Record the location and request field inspection before taking action."
    return "Detection logged for review. Confirm the object and its context."


def inject_custom_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

@property --angle { syntax:'<angle>'; inherits:false; initial-value:0deg; }
@property --n { syntax:'<integer>'; inherits:false; initial-value:0; }

:root {
  --bg:#060a12; --glass:rgba(255,255,255,.045); --glass-2:rgba(255,255,255,.08);
  --line:rgba(255,255,255,.09); --line-2:rgba(255,255,255,.18);
  --text:#e6ecf5; --muted:#8b98ad; --white:#fff;
  --mint:#34e3b0; --blue:#5b8cff; --violet:#9b6bff; --amber:#ffb547; --red:#ff5470;
  --grad:linear-gradient(110deg,#34e3b0,#5b8cff 55%,#9b6bff);
  --display:'Sora',sans-serif; --body:'Inter',sans-serif;
  --ease:cubic-bezier(.22,1,.36,1);
}

html, body, [class*="css"] { font-family:var(--body); }
.stApp { background:var(--bg); color:var(--text); overflow-x:hidden; }
.block-container { max-width:1440px; padding-top:1.2rem; padding-bottom:4rem; }
h1,h2,h3,h4 { font-family:var(--display) !important; color:var(--white) !important; letter-spacing:-.02em; }
p, label, li { color:var(--text); }
footer, #MainMenu { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }

/* ---------- Animated aurora background ---------- */
.stApp:before, .stApp:after {
  content:""; position:fixed; z-index:0; pointer-events:none; border-radius:50%; filter:blur(90px); opacity:.55;
}
.stApp:before { width:620px; height:620px; top:-180px; right:-120px;
  background:radial-gradient(circle,#2a6bff 0%,transparent 70%); animation:drift1 18s ease-in-out infinite alternate; }
.stApp:after { width:560px; height:560px; bottom:-200px; left:-140px;
  background:radial-gradient(circle,#18c9a0 0%,transparent 70%); animation:drift2 22s ease-in-out infinite alternate; }
@keyframes drift1 { to { transform:translate(-160px,140px) scale(1.18); } }
@keyframes drift2 { to { transform:translate(200px,-120px) scale(1.12); } }
.stApp [data-testid="stAppViewContainer"], section[data-testid="stSidebar"] { position:relative; z-index:1; }

/* ---------- Entrance choreography ---------- */
@keyframes rise { from { opacity:0; transform:translateY(22px); } to { opacity:1; transform:none; } }
@keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
.rise { opacity:0; animation:rise .9s var(--ease) forwards; animation-delay:calc(var(--i,0) * 90ms); }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg,rgba(14,22,38,.92),rgba(6,10,18,.96));
  backdrop-filter:blur(18px); border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] * { color:var(--text); }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { font-family:var(--display) !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background:var(--glass); border:1px solid var(--line-2); border-radius:12px; transition:border-color .25s, box-shadow .25s;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div:hover { border-color:var(--mint); box-shadow:0 0 0 3px rgba(52,227,176,.12); }
div[data-baseweb="slider"] div[role="slider"] { background:var(--mint) !important; box-shadow:0 0 0 6px rgba(52,227,176,.18); transition:box-shadow .2s; }
div[data-baseweb="slider"] div[role="slider"]:hover { box-shadow:0 0 0 10px rgba(52,227,176,.2); }
.side-brand { display:flex; align-items:center; gap:12px; padding:6px 0 4px; }
.side-logo {
  width:42px; height:42px; border-radius:13px; display:grid; place-items:center; font-size:21px;
  background:var(--grad); box-shadow:0 10px 30px rgba(91,140,255,.35); animation:float 4s ease-in-out infinite;
}
@keyframes float { 50% { transform:translateY(-4px); } }
.side-name { font:700 1.05rem var(--display); color:var(--white); }
.side-sub { font-size:.74rem; color:var(--muted); }
.note {
  border:1px solid rgba(255,181,71,.25); background:rgba(255,181,71,.07); color:#f3dcb0;
  padding:12px 14px; border-radius:12px; font-size:.82rem; line-height:1.55;
}

/* ---------- Top bar ---------- */
.topbar {
  display:flex; align-items:center; justify-content:space-between; padding:10px 14px 10px 18px; margin-bottom:18px;
  border:1px solid var(--line); border-radius:16px; background:var(--glass); backdrop-filter:blur(16px);
}
.topbar .name { font:700 1rem var(--display); color:var(--white); display:flex; gap:10px; align-items:center; }
.topbar .name span.leaf { display:inline-grid; place-items:center; width:28px; height:28px; border-radius:9px; background:var(--grad); font-size:15px; }
.live {
  display:inline-flex; align-items:center; gap:9px; padding:7px 14px; border-radius:999px; font-size:.78rem; font-weight:600;
  color:var(--mint); background:rgba(52,227,176,.08); border:1px solid rgba(52,227,176,.28);
}
.live i { width:8px; height:8px; border-radius:50%; background:var(--mint); animation:ping 2s ease-out infinite; }
@keyframes ping { 0% { box-shadow:0 0 0 0 rgba(52,227,176,.7);} 80%,100% { box-shadow:0 0 0 11px rgba(52,227,176,0);} }

/* ---------- Hero with animated border + radar ---------- */
.hero {
  position:relative; border-radius:26px; padding:48px 52px; margin-bottom:22px; overflow:hidden;
  display:grid; grid-template-columns:1.25fr .75fr; gap:30px; align-items:center;
  background:linear-gradient(135deg,rgba(18,30,52,.88),rgba(8,14,26,.92));
}
.hero:before {
  content:""; position:absolute; inset:0; border-radius:inherit; padding:1.5px; pointer-events:none;
  background:conic-gradient(from var(--angle), transparent 55%, #34e3b0 75%, #5b8cff 88%, transparent 100%);
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0); -webkit-mask-composite:xor;
  mask:linear-gradient(#000 0 0) content-box exclude, linear-gradient(#000 0 0);
  animation:spin 6s linear infinite;
}
@keyframes spin { to { --angle:360deg; } }
.hero h1 {
  font-size:clamp(2rem,3.9vw,3.4rem); line-height:1.08; margin:0 0 16px; font-weight:800 !important;
  background:linear-gradient(100deg,#fff 0%,#9fe9d4 35%,#8fb0ff 65%,#fff 100%); background-size:220% auto;
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; color:transparent !important;
  animation:shine 7s linear infinite;
}
@keyframes shine { to { background-position:220% center; } }
.hero p { color:#a9b7cc; font-size:1.02rem; line-height:1.7; max-width:56ch; margin:0 0 24px; }
.chips { display:flex; flex-wrap:wrap; gap:9px; }
.chips span {
  padding:8px 14px; border-radius:999px; font-size:.8rem; font-weight:500; color:#cfe0f5;
  background:var(--glass); border:1px solid var(--line); transition:transform .3s var(--ease), border-color .3s, background .3s;
}
.chips span:hover { transform:translateY(-3px); border-color:var(--mint); background:rgba(52,227,176,.09); }

.radar { position:relative; width:min(100%,300px); aspect-ratio:1; margin:0 auto; border-radius:50%;
  background:
    radial-gradient(circle,rgba(52,227,176,.18) 0 2px,transparent 3px),
    repeating-radial-gradient(circle,transparent 0 36px,rgba(52,227,176,.22) 37px 38px);
  border:1px solid rgba(52,227,176,.35); box-shadow:0 0 60px rgba(52,227,176,.12), inset 0 0 40px rgba(52,227,176,.08); }
.radar:before, .radar:after { content:""; position:absolute; background:rgba(52,227,176,.18); }
.radar:before { left:50%; top:0; bottom:0; width:1px; }
.radar:after { top:50%; left:0; right:0; height:1px; }
.sweep { position:absolute; inset:0; border-radius:50%;
  background:conic-gradient(from 0deg, rgba(52,227,176,0) 0deg, rgba(52,227,176,.5) 60deg, rgba(52,227,176,0) 61deg);
  animation:rot 4s linear infinite; }
@keyframes rot { to { transform:rotate(360deg); } }
.blip { position:absolute; width:10px; height:10px; border-radius:50%; background:var(--amber); box-shadow:0 0 14px var(--amber);
  animation:blip 4s ease-out infinite; opacity:0; }
.blip.b2 { background:var(--red); box-shadow:0 0 14px var(--red); }
@keyframes blip { 0%,18% { opacity:0; transform:scale(.4);} 24% { opacity:1; transform:scale(1.4);} 60% { opacity:.7; transform:scale(1);} 100% { opacity:0; } }

/* ---------- Stat cards with count-up ---------- */
.stats { display:grid; grid-template-columns:repeat(5,1fr); gap:14px; }
.stat {
  position:relative; padding:20px 22px; border-radius:18px; overflow:hidden;
  background:var(--glass); border:1px solid var(--line); backdrop-filter:blur(14px);
  transition:transform .4s var(--ease), border-color .3s, box-shadow .4s;
}
.stat:before { content:""; position:absolute; inset:0; opacity:0; transition:opacity .4s;
  background:radial-gradient(260px circle at 50% 0%, rgba(91,140,255,.22), transparent 70%); }
.stat:hover { transform:translateY(-6px); border-color:var(--line-2); box-shadow:0 22px 44px rgba(0,0,0,.35); }
.stat:hover:before { opacity:1; }
.stat .icon { font-size:1.25rem; margin-bottom:10px; display:block; }
.count { font:800 2.3rem/1 var(--display); color:var(--white); counter-reset:n var(--n); animation:cnt 1.5s var(--ease) forwards; position:relative; }
.count:after { content:counter(n); }
@keyframes cnt { from { --n:0; } to { --n:var(--t); } }
.stat .lbl { margin-top:8px; font-size:.82rem; color:var(--muted); position:relative; }
.stat.hot { border-color:rgba(255,84,112,.4); animation:hotpulse 2s ease-in-out infinite; }
@keyframes hotpulse { 50% { box-shadow:0 0 0 0 rgba(255,84,112,0), 0 0 30px rgba(255,84,112,.28); } }

/* ---------- Section headings ---------- */
.sec { display:flex; align-items:center; gap:14px; margin:36px 0 16px; }
.sec .bar { width:5px; height:28px; border-radius:4px; background:var(--grad); animation:grow .9s var(--ease) both; transform-origin:top; }
@keyframes grow { from { transform:scaleY(0); } }
.sec h3 { margin:0; font-size:1.4rem; }
.sec small { margin-left:auto; color:var(--muted); font-size:.84rem; }

/* ---------- Panels, alerts ---------- */
.panel { padding:20px 22px; border-radius:18px; background:var(--glass); border:1px solid var(--line); backdrop-filter:blur(14px); }
.panel-title { font:700 1.2rem var(--display); color:var(--white); margin-bottom:4px; }
.small-muted { color:var(--muted); font-size:.86rem; line-height:1.55; }
.empty { text-align:center; padding:38px 20px; border-radius:18px; border:1px dashed var(--line-2); background:var(--glass); }
.empty .em { font-size:2rem; display:block; animation:float 3.5s ease-in-out infinite; }
.empty b { display:block; font:700 1.1rem var(--display); color:var(--white); margin:6px 0 2px; }

.alert {
  position:relative; display:grid; grid-template-columns:auto 1fr; gap:16px; align-items:start; overflow:hidden;
  padding:18px 22px; margin-bottom:12px; border-radius:16px; background:var(--glass); border:1px solid var(--line);
  opacity:0; animation:slideIn .7s var(--ease) forwards; animation-delay:calc(var(--i,0) * 120ms);
  transition:transform .3s var(--ease), border-color .3s;
}
.alert:before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--sev); box-shadow:0 0 18px var(--sev); }
.alert:hover { transform:translateX(6px); border-color:var(--sev); }
@keyframes slideIn { from { opacity:0; transform:translateX(-40px); } to { opacity:1; transform:none; } }
.alert .sev { padding:6px 12px; border-radius:9px; font:700 .76rem var(--body); color:#0a0f18; background:var(--sev); }
.alert .head { font:700 1.08rem var(--display); color:var(--white); }
.alert .meta { color:var(--muted); font-size:.84rem; margin:3px 0 8px; }
.alert .body { font-size:.92rem; line-height:1.6; max-width:92ch; }
.sev-critical { --sev:#ff5470; animation:slideIn .7s var(--ease) forwards, crit 2s ease-in-out 1s infinite; }
@keyframes crit { 50% { box-shadow:0 0 28px rgba(255,84,112,.3); } }
.sev-high { --sev:#ffb547; } .sev-moderate { --sev:#f4de72; } .sev-low { --sev:#34e3b0; }

/* ---------- Images: reveal wipe + scan line ---------- */
div[data-testid="stImage"] { position:relative; border-radius:16px; overflow:hidden; border:1px solid var(--line-2); box-shadow:0 20px 50px rgba(0,0,0,.4); }
div[data-testid="stImage"] img { border-radius:0; animation:wipe 1.1s var(--ease) both; }
@keyframes wipe { from { clip-path:inset(0 100% 0 0); } to { clip-path:inset(0 0 0 0); } }
div[data-testid="stImage"]:after {
  content:""; position:absolute; top:0; bottom:0; width:70px; left:0; pointer-events:none; z-index:2;
  background:linear-gradient(90deg,transparent,rgba(52,227,176,.55),transparent);
  animation:scan 1.3s var(--ease) both;
}
@keyframes scan { from { transform:translateX(-80px); opacity:1; } to { transform:translateX(1400px); opacity:0; } }

/* ---------- Buttons, uploader, tables ---------- */
.stButton>button, .stDownloadButton>button {
  position:relative; overflow:hidden; border:none; border-radius:12px; padding:.6rem 1.1rem; font-weight:700; color:#06101a;
  background:var(--grad); background-size:180% 100%; transition:transform .25s var(--ease), box-shadow .3s, background-position .5s;
  box-shadow:0 8px 24px rgba(91,140,255,.28);
}
.stButton>button:hover, .stDownloadButton>button:hover {
  transform:translateY(-3px) scale(1.01); background-position:100% 0; color:#06101a; box-shadow:0 14px 34px rgba(52,227,176,.35);
}
.stButton>button:active, .stDownloadButton>button:active { transform:translateY(0) scale(.98); }
.stButton>button:after, .stDownloadButton>button:after {
  content:""; position:absolute; top:0; left:-120%; width:60%; height:100%;
  background:linear-gradient(100deg,transparent,rgba(255,255,255,.55),transparent); transform:skewX(-20deg); transition:left .7s;
}
.stButton>button:hover:after, .stDownloadButton>button:hover:after { left:140%; }
div[data-testid="stFileUploader"] {
  padding:14px; border-radius:18px; background:var(--glass); border:1.5px dashed rgba(52,227,176,.4);
  transition:border-color .3s, box-shadow .3s, transform .3s var(--ease);
}
div[data-testid="stFileUploader"]:hover { border-color:var(--mint); box-shadow:0 0 0 5px rgba(52,227,176,.1), 0 0 40px rgba(52,227,176,.12); transform:translateY(-2px); }
div[data-testid="stFileUploader"] section { background:transparent; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:16px; overflow:hidden; }
div[data-testid="stMetric"] { background:var(--glass); border:1px solid var(--line); border-radius:16px; padding:16px 20px; transition:transform .35s var(--ease); }
div[data-testid="stMetric"]:hover { transform:translateY(-4px); }
div[data-testid="stMetricLabel"] { color:var(--muted) !important; }
div[data-testid="stMetricValue"] { color:var(--white) !important; font-family:var(--display) !important; }
[data-testid="stAlert"] { border-radius:14px; backdrop-filter:blur(10px); }
div[data-testid="stSpinner"] i { border-top-color:var(--mint) !important; }
hr { border-color:var(--line) !important; }
.foot { text-align:center; margin-top:36px; color:var(--muted); font-size:.82rem; }

/* ---------- Keyboard focus + responsive + reduced motion ---------- */
button:focus-visible, input:focus-visible, [role="radio"]:focus-visible { outline:2px solid var(--mint) !important; outline-offset:3px; }
@media (max-width:980px) {
  .hero { grid-template-columns:1fr; padding:32px 24px; }
  .radar { display:none; }
  .stats { grid-template-columns:repeat(2,1fr); }
}
@media (prefers-reduced-motion: reduce) {
  *, *:before, *:after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; }
  .rise, .alert { opacity:1; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading computer-vision model...")
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def save_uploaded_weights(uploaded_weights) -> Optional[str]:
    if uploaded_weights is None:
        return None
    raw = uploaded_weights.getvalue()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"wildlife_safety_model_{digest}.pt")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(raw)
    return path


def boxes_are_near(box_a, box_b, image_width: int, image_height: int) -> bool:
    """Approximate image-space proximity. This does not confirm road presence."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    gap_x = max(bx1 - ax2, ax1 - bx2, 0)
    gap_y = max(by1 - ay2, ay1 - by2, 0)
    return gap_x <= image_width * 0.12 and gap_y <= image_height * 0.12


def detect_fire_smoke_color(rgb: np.ndarray) -> List[Tuple[str, float, List[int]]]:
    """Experimental colour/texture scan for flames and smoke. Returns (kind, confidence, [x1,y1,x2,y2])."""
    h, w = rgb.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1.0 else rgb
    sh, sw = small.shape[:2]
    area_total = float(sh * sw)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Flames: saturated red-orange-yellow that is also bright.
    fire_mask = cv2.inRange(hsv, (0, 110, 200), (35, 255, 255))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    # Smoke: low-saturation grey/white regions with soft internal variation.
    smoke_mask = cv2.inRange(hsv, (0, 0, 100), (179, 45, 225))
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

    found = []
    inv = 1.0 / scale

    def to_box(x, y, bw, bh):
        return [int(x * inv), int(y * inv), int((x + bw) * inv), int((y + bh) * inv)]

    for mask, kind, min_ratio in ((fire_mask, "fire", 0.002), (smoke_mask, "smoke", 0.03)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            ratio = (bw * bh) / area_total
            if ratio < min_ratio or ratio > 0.85:
                continue
            region_mask = mask[y:y + bh, x:x + bw] > 0
            fill = float(region_mask.mean())
            if fill < 0.30:
                continue
            roi = hsv[y:y + bh, x:x + bw]
            if kind == "fire":
                # Real flames contain a brighter, paler core and a wide brightness spread.
                hue = roi[..., 0][region_mask].astype(np.float32)
                if hue.size == 0 or float(hue.std()) < 3.0:  # flat single-colour objects are not flames
                    continue
                conf = min(0.92, 0.40 + 0.45 * fill + min(0.12, ratio * 4))
            else:
                v = roi[..., 2][region_mask].astype(np.float32)
                if v.size == 0:
                    continue
                spread = float(v.std())
                if spread < 7 or spread > 45:  # flat walls/sky or high-contrast clutter
                    continue
                conf = min(0.75, 0.28 + 0.40 * fill + min(0.10, ratio))
            found.append((kind, conf, to_box(x, y, bw, bh)))
    return found


def draw_tag(img, box, text, fill, text_color, width, height):
    x1, y1, x2, y2 = box
    thickness = max(2, int(round(min(width, height) / 250)))
    font_scale = max(0.5, min(width, height) / 1100.0)
    cv2.rectangle(img, (x1, y1), (x2, y2), fill, thickness)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1))
    top = max(0, y1 - th - 10)
    right = min(width - 1, x1 + tw + 10)
    bottom = min(height - 1, top + th + 10)
    cv2.rectangle(img, (x1, top), (right, bottom), fill, -1)
    cv2.putText(img, text, (x1 + 5, max(12, bottom - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                max(1, thickness - 1), cv2.LINE_AA)


def process_image(
    pil_image: Image.Image,
    model: YOLO,
    monitoring_mode: str,
    confidence: float,
    use_color_scan: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    rgb_image = np.array(pil_image.convert("RGB"))
    annotated = rgb_image.copy()
    height, width, _ = annotated.shape
    frame_area = float(width * height)
    results = model.predict(source=rgb_image, conf=0.10, verbose=False)
    detections: List[Dict[str, object]] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    animal_objects = []
    vehicle_boxes = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            detected_label = str(model.names.get(cls_id, cls_id)).lower().strip()
            conf_score = float(box.conf[0].item())
            coords = [int(round(v)) for v in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = coords
            coords = [
                max(0, min(x1, width - 1)), max(0, min(y1, height - 1)),
                max(0, min(x2, width - 1)), max(0, min(y2, height - 1)),
            ]

            if detected_label in VEHICLE_CLASSES:
                if conf_score >= 0.15:
                    vehicle_boxes.append((detected_label, conf_score, coords))
                continue

            if conf_score < confidence:
                continue

            display_label = "Wild Animal" if detected_label in ROAD_ANIMAL_CLASSES else detected_label
            category = classify_label(detected_label)
            animal_objects.append((detected_label, display_label, category, conf_score, coords))

    # Built-in fire/smoke scan: used only when the model itself found no fire/smoke,
    # so it never duplicates a trained model's detections.
    model_found_hazard = any(item[2] in HAZARD_CATEGORIES for item in animal_objects)
    if use_color_scan and not model_found_hazard and monitoring_mode in {"All Monitoring", "Environmental Hazards"}:
        for kind, score, box in detect_fire_smoke_color(rgb_image):
            if score < confidence:
                continue
            bx1, by1, bx2, by2 = [max(0, min(v, lim - 1)) for v, lim in zip(box, (width, height, width, height))]
            name = "Fire (colour scan)" if kind == "fire" else "Smoke (colour scan)"
            animal_objects.append((kind, name, classify_label(kind), score, [bx1, by1, bx2, by2]))

    if monitoring_mode in {"All Monitoring", "Wildlife & Road Safety"}:
        for vehicle_label, vehicle_conf, vcoords in vehicle_boxes:
            draw_tag(annotated, vcoords, f"{vehicle_label.upper()} {vehicle_conf:.0%}",
                     (255, 181, 71), (10, 15, 24), width, height)

    for detected_label, label, category, conf_score, coords in animal_objects:
        if monitoring_mode == "Forest Intrusion" and category != "Human Intrusion":
            continue
        if monitoring_mode == "Wildlife & Road Safety" and category != "Wildlife / Road Risk":
            continue
        if monitoring_mode == "Environmental Hazards" and category not in HAZARD_CATEGORIES:
            continue

        x1, y1, x2, y2 = coords
        box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
        proximity = compute_proximity(box_area, frame_area)
        color = color_for(category)

        road_context = False
        if category == "Wildlife / Road Risk":
            road_context = any(boxes_are_near(coords, vbox, width, height)
                               for _, _, vbox in vehicle_boxes)

        if category == "Wildlife / Road Risk":
            severity = "HIGH" if road_context else "LOW"
            if road_context:
                advisory = (
                    "POSSIBLE WILDLIFE ROAD-SAFETY ALERT: Wild Animal appears close to a detected vehicle. "
                    "Slow down, do not honk or approach, keep a safe distance, and allow an escape route. "
                    "Vehicle proximity is a visual clue; it does not prove the animal is on a road."
                )
                road_context_text = "Possible nearby vehicle"
            else:
                advisory = (
                    "Wild Animal detected. No nearby vehicle was detected by the model in this image. "
                    "Maintain a safe distance. Road presence is not confirmed."
                )
                road_context_text = "No nearby vehicle detected"
        else:
            severity = severity_for(category, proximity)
            advisory = get_advisory(category, label)
            if "colour scan" in label:
                advisory += " Found by colour analysis, not a trained model, so sunsets, lights or fog can trigger it. Confirm visually."
            road_context_text = "Not applicable"

        label_text = f"{label} {conf_score:.0%} | {severity}"
        if road_context:
            label_text += " | ROAD-RISK?"
        draw_tag(annotated, coords, label_text, color, (255, 255, 255), width, height)

        detections.append({
            "Timestamp": timestamp,
            "Category": category,
            "Object": label,
            "Confidence": f"{conf_score:.1%}",
            "Proximity": proximity,
            "Severity": severity,
            "Road Context": road_context_text,
            "Advisory": advisory,
        })

    return annotated, detections


def encode_png_bytes(rgb_array: np.ndarray) -> bytes:
    output = Image.fromarray(rgb_array)
    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    return buffer.getvalue()


def section(title: str, hint: str = "") -> None:
    hint_html = f"<small>{hint}</small>" if hint else ""
    st.markdown(f'<div class="sec"><span class="bar"></span><h3>{title}</h3>{hint_html}</div>', unsafe_allow_html=True)


def render_alert(detection: Dict[str, object], index: int) -> None:
    sev = str(detection["Severity"]).lower()
    st.markdown(
        f'<div class="alert sev-{sev}" style="--i:{index}">'
        f'<span class="sev">{detection["Severity"].title()}</span>'
        f'<div><div class="head">{detection["Category"]}</div>'
        f'<div class="meta">{detection["Object"]} at {detection["Confidence"]} confidence, proximity {detection["Proximity"].lower()}</div>'
        f'<div class="body">{detection["Advisory"]}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_stats(event_df: pd.DataFrame) -> None:
    has = not event_df.empty
    total = len(event_df)
    intrusion = int((event_df["Category"] == "Human Intrusion").sum()) if has else 0
    wildlife = int((event_df["Category"] == "Wildlife / Road Risk").sum()) if has else 0
    hazards = int(event_df["Category"].isin(list(HAZARD_CATEGORIES)).sum()) if has else 0
    critical = int((event_df["Severity"] == "CRITICAL").sum()) if has else 0

    def card(icon, value, label, extra=""):
        return (f'<div class="stat {extra}"><span class="icon">{icon}</span>'
                f'<div class="count" style="--t:{value}"></div><div class="lbl">{label}</div></div>')

    st.markdown(
        '<div class="stats rise" style="--i:3">'
        + card("📋", total, "Events this session")
        + card("🚷", intrusion, "Intrusion alerts")
        + card("🦌", wildlife, "Wildlife sightings")
        + card("🔥", hazards, "Environmental alerts")
        + card("⚠️", critical, "Critical events", "hot" if critical else "")
        + '</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_custom_css()
    if "event_log" not in st.session_state:
        st.session_state.event_log = []
    if "last_processed_key" not in st.session_state:
        st.session_state.last_processed_key = None

    st.markdown(
        """
        <div class="topbar rise" style="--i:0">
            <div class="name"><span class="leaf">🌿</span>Wildlife Safety</div>
            <div class="live"><i></i>System online</div>
        </div>
        <div class="hero rise" style="--i:1">
            <div>
                <h1>Protect wildlife. Prevent road incidents.</h1>
                <p>Upload camera evidence and get instant alerts for forest intrusion, animals near traffic and environmental hazards, each with a severity level and a clear next step.</p>
                <div class="chips">
                    <span>Object detection</span><span>Intrusion awareness</span><span>Road-risk checks</span><span>Event logging</span>
                </div>
            </div>
            <div class="radar">
                <div class="sweep"></div>
                <span class="blip" style="top:24%;left:62%;animation-delay:.6s"></span>
                <span class="blip b2" style="top:64%;left:30%;animation-delay:2.1s"></span>
                <span class="blip" style="top:70%;left:68%;animation-delay:3s"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            '<div class="side-brand"><div class="side-logo">🌿</div><div><div class="side-name">Wildlife Safety</div>'
            '<div class="side-sub">Monitoring console</div></div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        monitoring_mode = st.selectbox(
            "Monitoring module",
            ["All Monitoring", "Forest Intrusion", "Wildlife & Road Safety", "Environmental Hazards"],
        )
        confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
        st.divider()
        st.subheader("AI model")
        model_choice = st.radio("Model source", ["YOLOv8n (COCO baseline)", "Upload custom YOLO weights"], index=0)
        custom_weights = None
        if model_choice == "Upload custom YOLO weights":
            custom_weights = st.file_uploader("Upload model (.pt)", type=["pt"], help="Use custom-trained weights for fire, smoke, pests, disease or other specialist classes.")
        st.divider()
        use_color_scan = st.toggle(
            "Fire and smoke colour scan",
            value=True,
            help="Experimental. Finds flame-coloured and smoke-like regions without needing custom weights. Can give false alarms.",
        )
        st.divider()
        st.markdown("**Baseline detects:** people and common COCO animals")
        st.markdown("**Custom weights can add:** fire, smoke, pests, disease, lava and other trained classes")
        st.markdown('<div class="note">The baseline YOLOv8n COCO model does not recognize fire, smoke, volcanoes, pests or plant disease. Upload trained weights for those classes.</div>', unsafe_allow_html=True)

    model_path = "yolov8n.pt"
    if model_choice == "Upload custom YOLO weights":
        if custom_weights is None:
            st.warning("Upload your custom .pt model in the sidebar, or switch to the baseline model.")
            model_path = None
        else:
            model_path = save_uploaded_weights(custom_weights)

    render_stats(pd.DataFrame(st.session_state.event_log))

    section("Detection workspace", "Upload an image to begin")
    left, right = st.columns([1.05, 1.25], gap="large")
    with left:
        uploaded_file = st.file_uploader("Upload image evidence", type=["jpg", "jpeg", "png"], key="evidence")
        st.caption("JPG, JPEG or PNG.")
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Evidence image", use_container_width=True)

    with right:
        st.markdown(
            f'<div class="panel rise" style="--i:4"><div class="panel-title">{monitoring_mode}</div>'
            f'<p class="small-muted" style="margin:0 0 10px">Confidence threshold {confidence:.0%}</p>'
            f'<p style="margin:0">Each detection is categorized, given a severity level, and added to the session event log.</p></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        if monitoring_mode == "Environmental Hazards":
            st.info("Fire and smoke are found by the built-in colour scan (switch it on in the sidebar). Pests and disease still need a custom-trained model.")
        elif monitoring_mode == "Forest Intrusion":
            st.info("The baseline model detects the COCO 'person' class. A detection is an indicator, not proof of unauthorized entry.")
        elif monitoring_mode == "Wildlife & Road Safety":
            st.info("The baseline model recognizes common COCO animal classes. It cannot tell whether an animal is physically on a road.")

    if uploaded_file is not None and model_path is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            model = load_model(model_path)
            with st.spinner("Scanning image for wildlife, intrusion and hazards..."):
                annotated_img, detections = process_image(pil_image, model, monitoring_mode, confidence, use_color_scan)

            process_key = f"{uploaded_file.name}_{uploaded_file.size}_{monitoring_mode}_{confidence:.2f}_{model_path}_{use_color_scan}"
            if st.session_state.last_processed_key != process_key:
                st.session_state.event_log.extend(detections)
                st.session_state.last_processed_key = process_key

            section("Analysis results", "What the model found")
            a, b, c = st.columns(3)
            a.metric("Objects detected", len(detections))
            b.metric("High or critical", sum(d["Severity"] in {"HIGH", "CRITICAL"} for d in detections))
            c.metric("Image status", "Needs review" if any(d["Severity"] in {"HIGH", "CRITICAL"} for d in detections) else "Reviewed")

            st.write("")
            original_col, result_col = st.columns(2, gap="large")
            with original_col:
                st.markdown("#### Original")
                st.image(pil_image, use_container_width=True)
            with result_col:
                st.markdown("#### Detections")
                st.image(annotated_img, use_container_width=True)
                st.download_button(
                    "Download annotated image",
                    data=encode_png_bytes(annotated_img),
                    file_name=f"wildlife_safety_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                    mime="image/png",
                    use_container_width=True,
                )

            if detections:
                section("Alerts to review", f"{len(detections)} generated from this image")
                for i, detection in enumerate(detections):
                    render_alert(detection, i)
                st.dataframe(pd.DataFrame(detections), use_container_width=True, hide_index=True)
            else:
                st.success("Nothing matching this monitoring module was found above the confidence threshold.")
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            st.caption("Check that the model file is valid and that your installed Ultralytics and OpenCV versions are compatible.")

    section("Event log", "Everything recorded this session")
    if st.session_state.event_log:
        log_df = pd.DataFrame(st.session_state.event_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        csv_bytes = log_df.to_csv(index=False).encode("utf-8")
        download_col, clear_col = st.columns([1, 5])
        with download_col:
            st.download_button("Export CSV", data=csv_bytes, file_name="wildlife_safety_event_log.csv", mime="text/csv")
        with clear_col:
            if st.button("Clear session log"):
                st.session_state.event_log = []
                st.session_state.last_processed_key = None
                st.rerun()
    else:
        st.markdown('<div class="empty"><span class="em">🛰️</span><b>No events yet</b><span class="small-muted">Upload an image above and its detections will appear here.</span></div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="foot">Prototype decision-support system. Verify alerts in the field before taking operational action.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()