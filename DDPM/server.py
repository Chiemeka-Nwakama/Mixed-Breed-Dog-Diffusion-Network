# main server file - fastapi app that serves the UI and runs inference
# talks to inference.py for the actual model stuff

import os
import time
import threading
from typing import Optional, Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

from inference import load_model, generate_images, pil_to_base64_png



# shared dict so the SSE endpoint can read what the model is doing
_progress: dict = {"pct": 0, "step": 0, "total": 0, "active": False}
_progress_lock = threading.Lock()

# gets passed into generate_images() and called each diffusion step
# so we can stream progress to the browser without touching tqdm
class _ProgressCallback:
    """Injected into tqdm so we capture diffusion step progress."""
    def __init__(self, total):
        # reset progress state at the start of each generation
        with _progress_lock:
            _progress.update({"pct": 0, "step": 0, "total": total, "active": True})

    def update(self, n, step):
        with _progress_lock:
            _progress["step"] = step
            _progress["total"] = _progress["total"] or 1
            _progress["pct"] = min(int(step / _progress["total"] * 100), 99)

    def close(self):
        # mark as done so the SSE stream knows to send the final event
        with _progress_lock:
            _progress.update({"pct": 100, "active": False})

app = FastAPI(title="Dog Diffusion Generator")

MODEL = None
DEVICE = None

# pull config from env so we can override at runtime without rebuilding
IMG_SIZE = int(os.getenv("IMG_SIZE", "64"))
IN_CHANNELS = int(os.getenv("IN_CHANNELS", "3"))
DEFAULT_UPSCALE = int(os.getenv("UPSCALE_TO", "256"))

# maps class index -> breed name, matches the order the model was trained on
BREED_MAP: Dict[int, str] = {
    0: "Chihuahua",
    1: "Japanese spaniel",
    2: "Maltese dog",
    3: "Pekinese",
    4: "Shih-Tzu",
    5: "Blenheim spaniel",
    6: "Papillon",
    7: "Toy terrier",
    8: "Rhodesian ridgeback",
    9: "Afghan hound",
    10: "Basset",
    11: "Beagle",
    12: "Bloodhound",
    13: "Bluetick",
    14: "Black-and-tan coonhound",
    15: "Walker hound",
    16: "English foxhound",
    17: "Redbone",
    18: "Borzoi",
    19: "Irish wolfhound",
    20: "Italian greyhound",
    21: "Whippet",
    22: "Ibizan hound",
    23: "Norwegian elkhound",
    24: "Otterhound",
    25: "Saluki",
    26: "Scottish deerhound",
    27: "Weimaraner",
    28: "Staffordshire bullterrier",
    29: "American Staffordshire terrier",
    30: "Bedlington terrier",
    31: "Border terrier",
    32: "Kerry blue terrier",
    33: "Irish terrier",
    34: "Norfolk terrier",
    35: "Norwich terrier",
    36: "Yorkshire terrier",
    37: "Wire-haired fox terrier",
    38: "Lakeland terrier",
    39: "Sealyham terrier",
    40: "Airedale",
    41: "Cairn",
    42: "Australian terrier",
    43: "Dandie Dinmont",
    44: "Boston bull",
    45: "Miniature schnauzer",
    46: "Giant schnauzer",
    47: "Standard schnauzer",
    48: "Scotch terrier",
    49: "Tibetan terrier",
    50: "Silky terrier",
    51: "Soft-coated wheaten terrier",
    52: "West Highland white terrier",
    53: "Lhasa",
    54: "Flat-coated retriever",
    55: "Curly-coated retriever",
    56: "Golden retriever",
    57: "Labrador retriever",
    58: "Chesapeake Bay retriever",
    59: "German short-haired pointer",
    60: "Vizsla",
    61: "English setter",
    62: "Irish setter",
    63: "Gordon setter",
    64: "Brittany spaniel",
    65: "Clumber",
    66: "English springer",
    67: "Welsh springer spaniel",
    68: "Cocker spaniel",
    69: "Sussex spaniel",
    70: "Irish water spaniel",
    71: "Kuvasz",
    72: "Schipperke",
    73: "Groenendael",
    74: "Malinois",
    75: "Briard",
    76: "Kelpie",
    77: "Komondor",
    78: "Old English sheepdog",
    79: "Shetland sheepdog",
    80: "Collie",
    81: "Border collie",
    82: "Bouvier des Flandres",
    83: "Rottweiler",
    84: "German shepherd",
    85: "Doberman",
    86: "Miniature pinscher",
    87: "Greater Swiss Mountain dog",
    88: "Bernese mountain dog",
    89: "Appenzeller",
    90: "EntleBucher",
    91: "Boxer",
    92: "Bull mastiff",
    93: "Tibetan mastiff",
    94: "French bulldog",
    95: "Great Dane",
    96: "Saint Bernard",
    97: "Eskimo dog",
    98: "Malamute",
    99: "Siberian husky",
    100: "Affenpinscher",
    101: "Basenji",
    102: "Pug",
    103: "Leonberg",
    104: "Newfoundland",
    105: "Great Pyrenees",
    106: "Samoyed",
    107: "Pomeranian",
    108: "Chow",
    109: "Keeshond",
    110: "Brabancon griffon",
    111: "Pembroke",
    112: "Cardigan",
    113: "Toy poodle",
    114: "Miniature poodle",
    115: "Standard poodle",
    116: "Mexican hairless",
    117: "Dingo",
    118: "Dhole",
    119: "African hunting dog",
}


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>🐶 Dog Diffusion Generator</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #f5f0e8;
      --panel: #fffdf7;
      --border: #d6ccb4;
      --accent: #b85c2a;
      --accent2: #4a7c59;
      --text: #2a2318;
      --muted: #8a7d6a;
      --radius: 10px;
    }

    body {
      font-family: 'DM Mono', monospace;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem 1rem 4rem;
    }

    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }

    header h1 {
      font-family: 'Fraunces', serif;
      font-size: 2.8rem;
      font-weight: 300;
      letter-spacing: -0.5px;
      color: var(--accent);
    }

    header p {
      color: var(--muted);
      font-size: 0.8rem;
      margin-top: 0.3rem;
    }

    .layout {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 1.5rem;
      max-width: 1100px;
      margin: 0 auto;
      align-items: start;
    }

    @media (max-width: 720px) {
      .layout { grid-template-columns: 1fr; }
      header h1 { font-size: 2rem; }
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.5rem;
    }

    .panel h2 {
      font-family: 'Fraunces', serif;
      font-weight: 600;
      font-size: 1rem;
      margin-bottom: 1.2rem;
      padding-bottom: 0.6rem;
      border-bottom: 1px solid var(--border);
      color: var(--accent);
    }

    .field { margin-bottom: 1.1rem; }

    label {
      display: block;
      font-size: 0.72rem;
      color: var(--muted);
      margin-bottom: 0.35rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    input[type=number], select {
      width: 100%;
      padding: 0.5rem 0.7rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--bg);
      color: var(--text);
      font-family: 'DM Mono', monospace;
      font-size: 0.82rem;
      outline: none;
      transition: border-color 0.15s;
    }

    input[type=number]:focus, select:focus {
      border-color: var(--accent);
    }

    input[type=range] {
      width: 100%;
      accent-color: var(--accent);
      margin-top: 0.2rem;
    }

    .range-row {
      display: flex;
      align-items: center;
      gap: 0.7rem;
    }

    .range-val {
      min-width: 2.5rem;
      font-size: 0.82rem;
      color: var(--accent);
      text-align: right;
    }

    .tabs {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1.2rem;
    }

    .tab {
      flex: 1;
      padding: 0.45rem;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--bg);
      color: var(--muted);
      font-family: 'DM Mono', monospace;
      font-size: 0.72rem;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .tab.active {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }

    .tab-content { display: none; }
    .tab-content.active { display: block; }

    button#generate-btn {
      width: 100%;
      margin-top: 1rem;
      padding: 0.75rem;
      background: var(--accent);
      color: white;
      border: none;
      border-radius: var(--radius);
      font-family: 'Fraunces', serif;
      font-size: 1.1rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s, transform 0.1s;
    }

    button#generate-btn:hover { opacity: 0.88; }
    button#generate-btn:active { transform: scale(0.98); }
    button#generate-btn:disabled { opacity: 0.5; cursor: not-allowed; }

    #status {
      text-align: center;
      font-size: 0.75rem;
      color: var(--muted);
      margin-top: 0.8rem;
      min-height: 1.2em;
    }

    #status.error { color: #c0392b; }

    .results-panel h2 { margin-bottom: 1rem; }

    #image-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 0.9rem;
    }

    .img-card {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      background: var(--bg);
      position: relative;
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    .img-card img {
      width: 100%;
      display: block;
      image-rendering: pixelated;
    }

    .img-card a {
      display: block;
      text-align: center;
      padding: 0.4rem;
      font-size: 0.68rem;
      color: var(--accent2);
      text-decoration: none;
      border-top: 1px solid var(--border);
    }

    .img-card a:hover { text-decoration: underline; }

    .placeholder {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 220px;
      color: var(--muted);
      font-size: 0.8rem;
      gap: 0.5rem;
    }

    .placeholder span { font-size: 2.5rem; }

    .skeleton-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 0.9rem;
    }

    .skeleton-card {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      background: var(--panel);
    }

    .skeleton-img {
      width: 100%;
      aspect-ratio: 1;
      background: linear-gradient(90deg, var(--border) 25%, var(--bg) 50%, var(--border) 75%);
      background-size: 200% 100%;
      animation: shimmer 1.4s infinite;
    }

    .skeleton-bar {
      height: 24px;
      margin: 0.4rem;
      border-radius: 4px;
      background: linear-gradient(90deg, var(--border) 25%, var(--bg) 50%, var(--border) 75%);
      background-size: 200% 100%;
      animation: shimmer 1.4s infinite;
      animation-delay: 0.1s;
    }

    @keyframes shimmer {
      0%   { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }

    .spinner {
      display: inline-block;
      width: 1rem; height: 1rem;
      border: 2px solid rgba(184,92,42,0.2);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 0.4rem;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    #progress-wrap {
      display: none;
      margin-top: 1rem;
    }
    #progress-wrap.visible { display: block; }
    #progress-bar-track {
      width: 100%;
      height: 8px;
      background: var(--border);
      border-radius: 99px;
      overflow: hidden;
    }
    #progress-bar-fill {
      height: 100%;
      width: 0%;
      background: var(--accent);
      border-radius: 99px;
      transition: width 0.3s ease, background 0.3s;
    }
    #progress-label {
      font-size: 0.68rem;
      color: var(--muted);
      margin-top: 0.35rem;
      text-align: center;
    }

    #progress-wrap {
      display: none;
      margin-top: 1rem;
    }
    #progress-wrap.visible { display: block; }

    #progress-bar-track {
      width: 100%;
      height: 6px;
      background: var(--border);
      border-radius: 99px;
      overflow: hidden;
    }

    #progress-bar-fill {
      height: 100%;
      width: 0%;
      background: var(--accent);
      border-radius: 99px;
      transition: width 0.25s ease;
    }

    #progress-bar-fill.indeterminate {
      width: 40%;
      animation: indeterminate 1.2s ease-in-out infinite;
    }

    @keyframes indeterminate {
      0%   { transform: translateX(-100%); }
      100% { transform: translateX(300%); }
    }

    #progress-label {
      font-size: 0.68rem;
      color: var(--muted);
      margin-top: 0.35rem;
      text-align: center;
    }
  </style>
</head>
<body>

<header>
  <h1>🐶 Dog Diffusion</h1>
  <p>Conditional diffusion model · 120 breeds</p>
</header>

<div class="layout">

  <!-- LEFT: Controls -->
  <div>
    <div class="panel">
      <h2>Breed</h2>

      <div class="tabs">
        <button class="tab active" data-tab="single">Single</button>
        <button class="tab" data-tab="mix">Mix</button>
        <button class="tab" data-tab="random">Random</button>
      </div>

      <div class="tab-content active" id="tab-single">
        <div class="field">
          <label>Breed</label>
          <select id="breed_id">
            __BREED_OPTIONS__
          </select>
        </div>
      </div>

      <div class="tab-content" id="tab-mix">
        <div class="field">
          <label>Breed A</label>
          <select id="breed_a">
            __BREED_OPTIONS__
          </select>
        </div>
        <div class="field">
          <label>Breed B</label>
          <select id="breed_b">
            __BREED_OPTIONS__
          </select>
        </div>
        <div class="field">
          <label>Mix ratio (0 = all A, 1 = all B)</label>
          <div class="range-row">
            <input type="range" id="mix_ratio" min="0" max="1" step="0.05" value="0.5" />
            <span class="range-val" id="mix_ratio_val">0.50</span>
          </div>
        </div>
      </div>

      <div class="tab-content" id="tab-random">
        <p style="font-size:0.78rem;color:var(--muted);padding: 0.5rem 0;">
          No breed conditioning — the model generates freely.
        </p>
      </div>
    </div>

    <div class="panel" style="margin-top:1rem;">
      <h2>Generation</h2>

      <div class="field">
        <label>Number of images (1–32)</label>
        <input type="number" id="num_images" value="4" min="1" max="32" />
      </div>

      <div class="field">
        <label>Seed (0 = random each time)</label>
        <input type="number" id="seed" value="0" min="0" />
      </div>

      <div class="field">
        <label>Guidance scale</label>
        <div class="range-row">
          <input type="range" id="guidance_scale" min="1" max="15" step="0.5" value="5" />
          <span class="range-val" id="guidance_scale_val">5.0</span>
        </div>
      </div>

      <div class="field">
        <label>Upsample to (px)</label>
        <select id="upsample_to">
          <option value="64">64</option>
          <option value="128">128</option>
          <option value="256" selected>256</option>
          <option value="512">512</option>
          <option value="1024">1024</option>
        </select>
      </div>

      <button id="generate-btn">Generate</button>
      <div id="status"></div>
      <div id="progress-wrap">
        <div id="progress-bar-track"><div id="progress-bar-fill"></div></div>
        <div id="progress-label"></div>
      </div>
      <div id="progress-wrap">
        <div id="progress-bar-track">
          <div id="progress-bar-fill"></div>
        </div>
        <div id="progress-label">Starting…</div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Results -->
  <div class="panel results-panel">
    <h2>Results</h2>
    <div id="image-grid">
      <div class="placeholder">
        <span>🐾</span>
        Configure and generate to see images
      </div>
    </div>
  </div>

</div>

<script>
  // --- Tabs ---
  let activeTab = 'single';
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      activeTab = btn.dataset.tab;
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + activeTab).classList.add('active');
    });
  });

  // --- Range display ---
  function bindRange(id) {
    const el = document.getElementById(id);
    const val = document.getElementById(id + '_val');
    if (!el || !val) return;
    el.addEventListener('input', () => val.textContent = parseFloat(el.value).toFixed(2));
  }
  bindRange('mix_ratio');
  bindRange('guidance_scale');

  // --- Progress bar helpers ---
  const progressWrap  = document.getElementById('progress-wrap');
  const progressFill  = document.getElementById('progress-bar-fill');
  const progressLabel = document.getElementById('progress-label');
  let sseSource = null;

  function startProgress() {
    progressWrap.classList.add('visible');
    progressFill.style.width = '0%';
    progressFill.style.background = 'var(--accent)';
    progressLabel.textContent = 'Starting…';

    // open SSE connection - server streams real tqdm step counts
    if (sseSource) sseSource.close();
    sseSource = new EventSource('/progress');
    sseSource.onmessage = (e) => {
      if (e.data === 'done') { sseSource.close(); return; }
      const [pct, step, total] = e.data.split('|').map(Number);
      progressFill.style.width = pct + '%';
      progressLabel.textContent = total > 0
        ? `Step ${step} / ${total} — ${pct}%`
        : `${pct}%`;
    };
    sseSource.onerror = () => sseSource.close();
  }

  function showSkeletons(n) {
    // show placeholder shimmer cards in the results panel while generating
    const grid = document.getElementById('image-grid');
    const wrap = document.createElement('div');
    wrap.className = 'skeleton-grid';
    wrap.id = 'skeleton-wrap';
    for (let i = 0; i < n; i++) {
      const card = document.createElement('div');
      card.className = 'skeleton-card';
      card.innerHTML = '<div class="skeleton-img"></div><div class="skeleton-bar"></div>';
      wrap.appendChild(card);
    }
    grid.innerHTML = '';
    grid.appendChild(wrap);
  }

  function clearSkeletons() {
    const s = document.getElementById('skeleton-wrap');
    if (s) s.remove();
  }

  function finishProgress(success) {
    if (sseSource) { sseSource.close(); sseSource = null; }
    progressFill.style.width = '100%';
    progressFill.style.background = success ? 'var(--accent2)' : '#c0392b';
    progressLabel.textContent = success ? 'Done!' : 'Failed';
    setTimeout(() => {
      progressWrap.classList.remove('visible');
      progressFill.style.transition = 'none';
      progressFill.style.width = '0%';
      progressFill.style.background = 'var(--accent)';
      setTimeout(() => progressFill.style.transition = 'width 0.3s ease, background 0.3s', 50);
    }, 1500);
  }

  // --- Generate ---
  document.getElementById('generate-btn').addEventListener('click', async () => {
    const btn = document.getElementById('generate-btn');
    const status = document.getElementById('status');
    const grid = document.getElementById('image-grid');

    const num_images = parseInt(document.getElementById('num_images').value);
    const seed = parseInt(document.getElementById('seed').value);
    const guidance_scale = parseFloat(document.getElementById('guidance_scale').value);
    const upsample_to = parseInt(document.getElementById('upsample_to').value);

    let body = { num_images, seed, guidance_scale, upsample_to };

    if (activeTab === 'single') {
      body.breed_id = parseInt(document.getElementById('breed_id').value);
    } else if (activeTab === 'mix') {
      body.breed_a = parseInt(document.getElementById('breed_a').value);
      body.breed_b = parseInt(document.getElementById('breed_b').value);
      body.mix_ratio = parseFloat(document.getElementById('mix_ratio').value);
    }
    // random: no breed fields

    btn.disabled = true;
    status.className = '';
    status.innerHTML = '<span class="spinner"></span>Generating…';
    grid.innerHTML = '';
    startProgress();
    showSkeletons(num_images);

    try {
      const res = await fetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      finishProgress(true);
      clearSkeletons();
      status.textContent = `✓ ${data.images.length} image${data.images.length !== 1 ? 's' : ''} generated`;

      data.images.forEach((b64, i) => {
        const src = 'data:image/png;base64,' + b64;
        const card = document.createElement('div');
        card.className = 'img-card';
        card.style.animationDelay = (i * 50) + 'ms';

        const img = document.createElement('img');
        img.src = src;
        img.alt = 'Generated dog ' + (i + 1);

        const link = document.createElement('a');
        link.href = src;
        link.download = `dog_${i + 1}.png`;
        link.textContent = '↓ Download';

        card.appendChild(img);
        card.appendChild(link);
        grid.appendChild(card);
      });

    } catch (e) {
      finishProgress(false);
      clearSkeletons();
      status.className = 'error';
      status.textContent = '✗ ' + e.message;
      grid.innerHTML = '<div class="placeholder"><span>⚠️</span>' + e.message + '</div>';
    } finally {
      btn.disabled = false;
    }
  });
</script>

</body>
</html>

"""

class GenerateRequest(BaseModel):
    num_images: int = Field(4, ge=1, le=32)
    seed: int = 0
    guidance_scale: float = 5.0

    breed_id: Optional[int] = None
    breed_a: Optional[int] = None
    breed_b: Optional[int] = None
    mix_ratio: float = Field(0.5, ge=0.0, le=1.0)

    upsample_to: int = Field(DEFAULT_UPSCALE, ge=64, le=1024)


# load the model once at startup so we're not reloading it every request
@app.on_event("startup")
def startup():
    global MODEL, DEVICE
    # use GPU if available, otherwise fall back to CPU (will be slow)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.getenv("CKPT_PATH", "/checkpoints/conditional/stage2_conditional.pth")
    config_path = os.getenv("CONFIG_PATH", "/app/config.yml")
    MODEL = load_model(ckpt_path=ckpt_path, config_path=config_path, device=DEVICE)



# browser connects to this when generation starts and gets live step updates
# uses SSE (server-sent events) so we dont need websockets
@app.get("/progress")
def progress_stream():
    """SSE endpoint — browser subscribes and gets step updates."""
    def event_gen():
        while True:
            with _progress_lock:
                p = dict(_progress)
            data = f"data: {p['pct']}|{p['step']}|{p['total']}|{int(p['active'])}\n\n"
            yield data
            if not p["active"] and p["pct"] == 100:
                # send one final 100% then a done signal
                yield "data: done\n\n"
                break
            time.sleep(0.25)
    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# just stops the browser from spamming 404s in the logs
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model_loaded": MODEL is not None}


@app.get("/", response_class=HTMLResponse)
def index():
    options = "\n".join(
        f'<option value="{k}">{k}: {v}</option>'
        for k, v in sorted(BREED_MAP.items())
    )
    html = HTML_TEMPLATE.replace("__BREED_OPTIONS__", options)
    return HTMLResponse(html)


@app.post("/generate")
def generate(req: GenerateRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # cant mix single breed mode and mix mode at the same time
    if req.breed_id is not None and (req.breed_a is not None or req.breed_b is not None):
        raise HTTPException(status_code=400, detail="Provide either breed_id OR (breed_a, breed_b).")

    # helper to validate breed ids before we pass them to the model
    def _check(i: int, name: str):
        if i not in BREED_MAP:
            raise HTTPException(status_code=400, detail=f"{name} is invalid: {i}")

    if req.breed_id is not None:
        _check(req.breed_id, "breed_id")
    if req.breed_a is not None:
        _check(req.breed_a, "breed_a")
    if req.breed_b is not None:
        _check(req.breed_b, "breed_b")

    # 1000 steps matches the default T in the config, change if you tweaked it
    cb = _ProgressCallback(total=1000)
    imgs = generate_images(
        model=MODEL,
        device=DEVICE,
        num_images=req.num_images,
        seed=req.seed,
        guidance_scale=req.guidance_scale,
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        breed_id=req.breed_id,
        breed_a=req.breed_a,
        breed_b=req.breed_b,
        mix_ratio=req.mix_ratio,
        progress_callback=cb,
    )
    cb.close()

    # upsample after generation since the model outputs at IMG_SIZE (64px by default)
    up = int(req.upsample_to or 0)
    if up > 0:
        up = max(64, min(1024, up))  # clamp just in case
        imgs = [im.resize((up, up), resample=Image.BICUBIC) if im.size[0] != up else im for im in imgs]

    return {"images": [pil_to_base64_png(im) for im in imgs]}