import html
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from core.tutor import ask_tutor
from core.pdf_processor import extract_pdf_text, create_chunks
from core.rag import StudyRAG

from core.study_tools import (
    generate_summary,
    generate_notes,
    generate_quiz
)


# Make the Gemini key available to the SDK
try:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass  # fall back to an existing GEMINI_API_KEY environment variable


APP_NAME = "Smart Study AI"


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# HELPERS
# ==========================================

NAV_PAGES = ["Home", "AI Tutor", "Ask Your PDF", "Study Tools", "Focus Timer"]

TUTOR_STYLES = {
    "Standard": "",
    "Simple (like I'm 12)": (
        "Explain in very simple words, using an everyday analogy. "
    ),
    "Exam-focused": (
        "Answer the way a good exam answer is written: a short definition, "
        "the key points, then one brief example. "
    ),
    "Step-by-step": "Explain this in clear numbered steps. ",
}

TUTOR_PROMPTS = [
    "Explain Python functions with an example.",
    "What is the difference between a list and a tuple?",
    "Explain how photosynthesis works, step by step.",
]

PDF_PROMPTS = [
    "Summarize the main ideas of this document.",
    "List the key terms and what they mean.",
    "What are the most important points to remember for an exam?",
]

_ICON_PATHS = {
    "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "file": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>'
    ),
    "notes": (
        '<path d="M11 6h10"/><path d="M11 12h10"/><path d="M11 18h10"/>'
        '<path d="m3 6 1.5 1.5L7 5"/><path d="m3 12 1.5 1.5L7 11"/><path d="m3 18 1.5 1.5L7 17"/>'
    ),
    "quiz": (
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>'
    ),
    # Graduation cap with a sparkle (sidebar logo)
    "logo": (
        '<path d="M12 5 1.5 10.5 12 16l10.5-5.5z"/>'
        '<path d="M6 13.2v4.3c0 1.5 2.7 3 6 3s6-1.5 6-3v-4.3"/>'
        '<path d="M22.5 10.5v6"/>'
        '<path d="M18.5 1.5v3"/><path d="M17 3h3"/>'
    ),
}


def icon(name):
    return (
        '<svg viewBox="0 0 24 24" aria-hidden="true">'
        + _ICON_PATHS[name]
        + "</svg>"
    )


def render(markup):
    """Render an HTML snippet. Lines are joined so Markdown never
    mistakes indented HTML for a code block."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def page_header(title, subtitle):
    render(f"""
    <div class="page-title">{title}</div>
    <div class="page-sub">{subtitle}</div>
    """)


def empty_state(title, text):
    render(f"""
    <div class="empty">
        <div class="empty-title">{title}</div>
        <div class="empty-text">{text}</div>
    </div>
    """)


def doc_bar():
    name = html.escape(st.session_state.document_name or "")
    pages = st.session_state.doc_pages
    chunks = st.session_state.doc_chunks
    render(f"""
    <div class="docbar">
        {icon("file")}
        <div>
            <div class="docbar-name">{name}</div>
            <div class="docbar-meta">{pages} pages, {chunks} chunks indexed</div>
        </div>
    </div>
    """)


def go_to(page):
    st.session_state.nav = page


def set_pending(question):
    st.session_state.pending_q = question


def clear_chat(title):
    st.session_state.chat_histories[title] = []


def transcript_text(title, history):
    lines = [f"{APP_NAME} - {title}", ""]
    for message in history:
        who = "You" if message["role"] == "user" else APP_NAME
        lines += [f"{who}:", message["content"], ""]
    return "\n".join(lines)


def chat_toolbar(key, title, history):
    """Download / clear buttons shown under a chat once it has messages."""
    if not history:
        return

    c1, c2, _ = st.columns([1, 1, 2])

    with c1:
        st.download_button(
            "Download chat",
            data=transcript_text(title, history),
            file_name=f"{key}_chat.txt",
            mime="text/plain",
            key=f"dl_{key}",
            use_container_width=True
        )

    with c2:
        st.button(
            "Clear chat",
            key=f"clear_{key}",
            on_click=clear_chat,
            args=(title,),
            use_container_width=True
        )


def suggestion_buttons(prompts, key_prefix):
    cols = st.columns(len(prompts))

    for i, text in enumerate(prompts):
        with cols[i]:
            st.button(
                text,
                key=f"{key_prefix}_{i}",
                on_click=set_pending,
                args=(text,),
                use_container_width=True
            )


TIMER_HTML = """
<style>
  * { box-sizing: border-box; font-family: 'DM Sans', system-ui, sans-serif; }
  body { margin: 0; }
  .wrap { position: relative; overflow: hidden; border-radius: 20px;
          padding: 24px 20px 26px; text-align: center; color: #fff; min-height: 500px;
          background: linear-gradient(130deg, #131B33, #1D2C6B, #2748E8, #1D2747);
          background-size: 300% 300%; animation: bg 14s ease infinite; }
  @keyframes bg { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
  .bub { position: absolute; bottom: -100px; border-radius: 50%;
         background: rgba(255,255,255,0.08); animation: rise linear infinite; }
  @keyframes rise { 0% { transform: translateY(0); opacity: 0.9; } 100% { transform: translateY(-640px); opacity: 0; } }
  .in { position: relative; z-index: 2; }
  .modes { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
  .modes button { border: 1px solid rgba(255,255,255,0.3); background: rgba(255,255,255,0.08);
          color: #E8ECF8; border-radius: 999px; padding: 7px 16px; font-weight: 600;
          font-size: 14px; cursor: pointer; transition: background 0.2s; }
  .modes button:hover { background: rgba(255,255,255,0.2); }
  .modes button.on { background: #FFD93D; color: #131B33; border-color: #FFD93D; }
  .ringbox { position: relative; width: 250px; height: 250px; margin: 22px auto 18px; }
  .ring { width: 250px; height: 250px; transition: transform 0.3s; }
  .ring.run { animation: breathe 2.4s ease-in-out infinite;
              filter: drop-shadow(0 0 14px rgba(255,217,61,0.55)); }
  @keyframes breathe { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.035); } }
  .track { fill: none; stroke: rgba(255,255,255,0.14); stroke-width: 10; }
  .prog { fill: none; stroke: url(#g); stroke-width: 10; stroke-linecap: round;
          transition: stroke-dashoffset 0.5s linear; }
  .center { position: absolute; inset: 0; display: flex; flex-direction: column;
            align-items: center; justify-content: center; }
  #t { font-size: 62px; font-weight: 700; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
  #lbl { font-size: 14px; color: #C5CEE6; margin-top: 2px; letter-spacing: 0.08em; text-transform: uppercase; }
  .wrap.done #t { animation: pop 0.6s ease 4; }
  @keyframes pop { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.18); color: #FFD93D; } }
  .ripple { position: absolute; inset: 0; border-radius: 50%; border: 3px solid #FFD93D;
            opacity: 0; pointer-events: none; }
  .wrap.done .ripple { animation: rip 1.6s ease-out 3; }
  .wrap.done .ripple:nth-of-type(2) { animation-delay: 0.5s; }
  @keyframes rip { 0% { transform: scale(0.8); opacity: 0.9; } 100% { transform: scale(1.7); opacity: 0; } }
  .ctl { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
  .ctl button { border-radius: 10px; padding: 10px 24px; font-weight: 700; font-size: 15px;
          cursor: pointer; border: 1px solid #FFD93D; background: #FFD93D; color: #131B33; }
  .ctl button.ghost { background: rgba(255,255,255,0.1); color: #fff; border-color: rgba(255,255,255,0.35); }
  .ctl button:hover { filter: brightness(1.08); }
  #msg { margin-top: 16px; font-size: 14px; color: #C5CEE6; min-height: 20px; }
  .cf { position: absolute; top: -14px; width: 9px; height: 14px; z-index: 3; border-radius: 2px;
        animation: fall linear forwards; pointer-events: none; }
  @keyframes fall { to { transform: translateY(620px) rotate(720deg); opacity: 0.9; } }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
</style>
<div class="wrap" id="wrap">
  <div class="in">
    <div class="modes">
      <button data-m="focus" class="on">Focus</button>
      <button data-m="short">Short break</button>
      <button data-m="long">Long break</button>
    </div>
    <div class="ringbox">
      <div class="ripple"></div><div class="ripple"></div>
      <svg class="ring" id="rsvg" viewBox="0 0 250 250">
        <defs>
          <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#FFD93D"/>
            <stop offset="100%" stop-color="#8FA8FF"/>
          </linearGradient>
        </defs>
        <circle class="track" cx="125" cy="125" r="108"/>
        <circle class="prog" id="ring" cx="125" cy="125" r="108" transform="rotate(-90 125 125)"/>
      </svg>
      <div class="center"><div id="t">25:00</div><div id="lbl">Focus</div></div>
    </div>
    <div class="ctl">
      <button id="go">Start</button>
      <button id="rs" class="ghost">Reset</button>
      <button id="snd" class="ghost">Sound: on</button>
      <button id="test" class="ghost">Test sound</button>
    </div>
    <div id="msg">Sessions completed: 0</div>
  </div>
</div>
<script>
  const D = { focus: __FOCUS__ * 60, short: __SHORT__ * 60, long: __LONG__ * 60 };
  const NAMES = { focus: "Focus", short: "Short break", long: "Long break" };
  let mode = "focus", left = D.focus, timer = null, endAt = 0, done = 0, soundOn = true, ctx = null;
  const $ = (id) => document.getElementById(id);
  const wrap = $("wrap"), ring = $("ring"), rsvg = $("rsvg");
  const C = 2 * Math.PI * 108;
  ring.style.strokeDasharray = C;

  function fmt(s) {
    return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  }
  function draw() {
    $("t").textContent = fmt(left);
    $("lbl").textContent = NAMES[mode];
    ring.style.strokeDashoffset = C * (left / D[mode]);
  }
  function unlock() {
    try {
      if (!ctx) { ctx = new (window.AudioContext || window.webkitAudioContext)(); }
      if (ctx.state === "suspended") { ctx.resume(); }
    } catch (e) {}
  }
  function chime() {
    if (!soundOn) { return; }
    unlock();
    if (!ctx) { return; }
    const notes = [784, 988, 1175, 1568];
    const t0 = ctx.currentTime + 0.05;
    for (let r = 0; r < 3; r++) {
      notes.forEach((f, i) => {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.type = "sine";
        o.frequency.value = f;
        o.connect(g);
        g.connect(ctx.destination);
        const s = t0 + r * 1.3 + i * 0.22;
        g.gain.setValueAtTime(0.0001, s);
        g.gain.exponentialRampToValueAtTime(0.45, s + 0.03);
        g.gain.exponentialRampToValueAtTime(0.0001, s + 0.7);
        o.start(s);
        o.stop(s + 0.75);
      });
    }
  }
  function confetti() {
    const cols = ["#FFD93D", "#8FA8FF", "#FFFFFF", "#6EE7B0", "#FF8FA3"];
    for (let i = 0; i < 44; i++) {
      const c = document.createElement("div");
      c.className = "cf";
      c.style.left = Math.random() * 100 + "%";
      c.style.background = cols[i % cols.length];
      c.style.animationDuration = (2 + Math.random() * 2.2) + "s";
      c.style.animationDelay = (Math.random() * 0.6) + "s";
      wrap.appendChild(c);
      setTimeout(() => c.remove(), 5500);
    }
  }
  function stop() {
    clearInterval(timer);
    timer = null;
    $("go").textContent = "Start";
    rsvg.classList.remove("run");
  }
  function finish() {
    stop();
    wrap.classList.add("done");
    chime();
    confetti();
    if (mode === "focus") { done += 1; }
    $("msg").textContent = "Time is up! Sessions completed: " + done +
      (mode === "focus" ? ". Take a break." : ". Ready for the next round?");
    setTimeout(() => wrap.classList.remove("done"), 6000);
  }
  function tick() {
    left = Math.max(0, Math.round((endAt - Date.now()) / 1000));
    draw();
    if (left === 0) { finish(); }
  }
  $("go").onclick = () => {
    unlock();
    wrap.classList.remove("done");
    if (timer) { stop(); return; }
    if (left === 0) { left = D[mode]; }
    endAt = Date.now() + left * 1000;
    timer = setInterval(tick, 500);
    $("go").textContent = "Pause";
    rsvg.classList.add("run");
  };
  $("rs").onclick = () => { unlock(); stop(); wrap.classList.remove("done"); left = D[mode]; draw(); };
  $("snd").onclick = () => {
    unlock();
    soundOn = !soundOn;
    $("snd").textContent = "Sound: " + (soundOn ? "on" : "off");
  };
  $("test").onclick = () => { unlock(); chime(); };
  document.querySelectorAll(".modes button").forEach((b) => {
    b.onclick = () => {
      unlock();
      document.querySelectorAll(".modes button").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      stop();
      wrap.classList.remove("done");
      mode = b.dataset.m;
      left = D[mode];
      draw();
    };
  });
  for (let i = 0; i < 14; i++) {
    const b = document.createElement("div");
    const size = 20 + Math.random() * 70;
    b.className = "bub";
    b.style.width = b.style.height = size + "px";
    b.style.left = Math.random() * 100 + "%";
    b.style.animationDuration = (9 + Math.random() * 10) + "s";
    b.style.animationDelay = (-Math.random() * 14) + "s";
    wrap.insertBefore(b, wrap.firstChild);
  }
  draw();
</script>
"""


# ==========================================
# STYLES
# ==========================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

:root {
    --ink: #131B33;
    --ink2: #1D2747;
    --paper: #F4F6FA;
    --surface: #FFFFFF;
    --line: #E0E5F0;
    --text: #172036;
    --muted: #5B6783;
    --cobalt: #2748E8;
    --cobalt-d: #1D38C0;
    --marker: #FFD93D;
    --marker-soft: #FFF4BF;
    --ok: #1F8A5B;
}

/* ---------- Base ---------- */
.stApp {
    background: var(--paper);
    color: var(--text);
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp button,
.stApp input, .stApp textarea {
    font-family: 'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp p, .stApp li { color: var(--text); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDeployButton"] { display: none; }
.block-container {
    max-width: 1080px;
    padding-top: 2.4rem;
    padding-bottom: 6rem;
}
.stApp :focus-visible { outline: 2px solid var(--cobalt); outline-offset: 2px; }

/* ---------- Page headers ---------- */
.page-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 36px;
    font-weight: 600;
    line-height: 1.15;
    letter-spacing: -0.015em;
    color: var(--text);
    margin: 0 0 8px;
}
.page-sub {
    font-size: 16px;
    line-height: 1.6;
    color: var(--muted);
    max-width: 62ch;
    margin: 0 0 26px;
}
.section-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 24px;
    font-weight: 600;
    color: var(--text);
    margin: 44px 0 16px;
}

/* ---------- Home hero ---------- */
.hero-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: clamp(36px, 5vw, 54px);
    font-weight: 600;
    line-height: 1.08;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 10px 0 18px;
}
.hero-sub {
    font-size: 17px;
    line-height: 1.65;
    color: var(--muted);
    max-width: 50ch;
    margin: 0 0 26px;
}
.steps {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 26px 26px 6px;
}
.steps-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 18px;
}
.step { display: flex; gap: 14px; padding-bottom: 20px; }
.step-n {
    flex: 0 0 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--ink);
    color: #fff;
    font-size: 13px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}
.step-t { font-weight: 700; font-size: 15px; color: var(--text); }
.step-d { font-size: 14px; line-height: 1.5; color: var(--muted); margin-top: 2px; }

/* ---------- Capabilities ---------- */
.cap-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    overflow: hidden;
}
.cap { padding: 24px 22px; border-right: 1px solid var(--line); }
.cap:last-child { border-right: none; }
.cap svg, .docbar svg {
    width: 24px;
    height: 24px;
    stroke: var(--cobalt);
    fill: none;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.cap svg { margin-bottom: 14px; }
.cap-t { font-weight: 700; font-size: 15px; color: var(--text); margin-bottom: 6px; }
.cap-d { font-size: 14px; line-height: 1.55; color: var(--muted); }
@media (max-width: 900px) {
    .cap-grid { grid-template-columns: repeat(2, 1fr); }
    .cap:nth-child(2) { border-right: none; }
    .cap:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
}

.spec-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    border-top: 1px solid var(--line);
    padding-top: 18px;
}
.spec-k { font-size: 13px; color: var(--muted); }
.spec-v { font-size: 15px; font-weight: 600; color: var(--text); margin-top: 2px; }
@media (max-width: 900px) { .spec-grid { grid-template-columns: repeat(2, 1fr); } }

/* ---------- Document bar, empty states ---------- */
.docbar {
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 4px solid var(--marker);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 22px;
}
.docbar svg { flex: 0 0 22px; width: 22px; height: 22px; }
.docbar-name { font-weight: 700; font-size: 15px; color: var(--text); word-break: break-all; }
.docbar-meta { font-size: 13px; color: var(--muted); }

.empty {
    background: var(--surface);
    border: 1px dashed #BAC4DA;
    border-radius: 16px;
    padding: 44px 28px;
    text-align: center;
}
.empty-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 6px;
}
.empty-text { font-size: 15px; line-height: 1.6; color: var(--muted); max-width: 46ch; margin: 0 auto; }

/* ---------- Source passages ---------- */
.source { padding: 14px 0; border-bottom: 1px solid var(--line); }
.source:last-child { border-bottom: none; }
.source-head { display: flex; justify-content: space-between; margin-bottom: 8px; }
.source-page { font-weight: 700; font-size: 14px; color: var(--text); }
.source-score { font-size: 13px; color: var(--muted); }
.source-text { font-size: 14px; line-height: 1.85; color: var(--text); }
.source-text mark {
    background: linear-gradient(transparent 55%, var(--marker) 55%);
    color: inherit;
    -webkit-box-decoration-break: clone;
    box-decoration-break: clone;
}

/* ---------- Tools ---------- */
.tool-desc { font-size: 15px; line-height: 1.6; color: var(--muted); max-width: 62ch; margin: 16px 0 14px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs button[role="tab"] { height: 46px; padding: 0 14px; color: var(--muted); }
.stTabs button[role="tab"] p { color: inherit; font-size: 15px; font-weight: 600; }
.stTabs button[role="tab"][aria-selected="true"] { color: var(--cobalt); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--cobalt); }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    background: var(--cobalt);
    border: 1px solid var(--cobalt);
    border-radius: 10px;
    padding: 0.55rem 1.1rem;
    font-weight: 600;
    transition: background 0.15s;
}
.stButton > button p, .stDownloadButton > button p { color: #fff !important; font-weight: 600; }
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--cobalt-d);
    border-color: var(--cobalt-d);
}
.stDownloadButton > button { background: var(--surface); }
.stDownloadButton > button p { color: var(--cobalt) !important; }
.stDownloadButton > button:hover { background: #EEF1FF; border-color: var(--cobalt); }

[class*="st-key-prompt_"] button, [class*="st-key-cta_tutor"] button,
[class*="st-key-clear_"] button {
    background: var(--surface);
    border: 1px solid var(--line);
    color: var(--text);
    text-align: left;
    justify-content: flex-start;
}
[class*="st-key-prompt_"] button p, [class*="st-key-cta_tutor"] button p,
[class*="st-key-clear_"] button p {
    color: var(--text) !important;
    font-weight: 500;
}
[class*="st-key-prompt_"] button:hover, [class*="st-key-cta_tutor"] button:hover,
[class*="st-key-clear_"] button:hover {
    background: #EEF1FF;
    border-color: var(--cobalt);
}

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 18px;
}
[data-testid="stChatInput"] { border-radius: 14px; border: 1px solid #C5CEE3; background: var(--surface); }
[data-testid="stBottom"] > div { background: var(--paper); }

/* ---------- Misc widgets ---------- */
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--ink); border-right: none; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] small, [data-testid="stSidebar"] li { color: #C5CEE6; }
[data-testid="stSidebar"] :focus-visible { outline-color: var(--marker); }

.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 22px; }
.brand-mark {
    flex: 0 0 44px;
    height: 44px;
    border-radius: 13px;
    background: linear-gradient(135deg, #FFD93D 0%, #FFB347 55%, #FF8FA3 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 18px rgba(255, 217, 61, 0.35);
    transform: rotate(-4deg);
}
.brand-mark svg {
    width: 27px;
    height: 27px;
    stroke: var(--ink);
    fill: none;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.brand-name { font-family: 'Fraunces', Georgia, serif; font-size: 20px; font-weight: 600; color: #fff; line-height: 1.1; }
.brand-sub { font-size: 12px; color: #93A0C4; margin-top: 2px; }
.side-label { font-size: 13px; font-weight: 700; color: #93A0C4; margin: 22px 0 8px; }

[data-testid="stSidebar"] [role="radiogroup"] { display: flex; flex-direction: column; gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    width: 100%;
    padding: 10px 12px;
    border-radius: 10px;
    border-left: 3px solid transparent;
    cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--ink2); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: var(--ink2);
    border-left-color: var(--marker);
}
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 15px; font-weight: 500; color: #C5CEE6; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: #fff; font-weight: 600; }

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--ink2);
    border: 1px solid #4B5B8F;
    border-radius: 10px;
}
[data-testid="stSidebar"] [data-baseweb="select"] * { color: #fff; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #C5CEE6; }

[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: var(--ink2);
    border: 1px dashed #4B5B8F;
    border-radius: 12px;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #C5CEE6; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: transparent;
    border: 1px solid #5A6BA0;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * { color: #fff; }
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * { color: #E8ECF8; }

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: var(--marker);
    border-color: var(--marker);
}
[data-testid="stSidebar"] .stButton > button p { color: var(--ink) !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: #FFE46B; border-color: #FFE46B; }

.doc-card {
    background: var(--ink2);
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 14px;
}
.doc-card-title { font-size: 14px; font-weight: 700; color: #fff; word-break: break-all; }
.doc-card-meta { font-size: 13px; color: #93A0C4; margin-top: 3px; }
.doc-card-state { font-size: 13px; font-weight: 600; color: #6EE7B0; margin-top: 8px; }
.engine { margin-top: 28px; padding-top: 18px; border-top: 1px solid #2A3558; }
.engine-row { font-size: 13px; color: #93A0C4; line-height: 1.8; }


/* ---------- Motion ---------- */
.stApp { isolation: isolate; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent; }
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    z-index: -1;
    border-radius: 50%;
    filter: blur(70px);
    pointer-events: none;
}
.stApp::before {
    width: 440px; height: 440px; top: -120px; right: -60px;
    background: radial-gradient(circle, #B9C6FF, transparent 70%);
    opacity: 0.55;
    animation: drift1 18s ease-in-out infinite alternate;
}
.stApp::after {
    width: 400px; height: 400px; bottom: -140px; left: 28%;
    background: radial-gradient(circle, #FFE58A, transparent 70%);
    opacity: 0.5;
    animation: drift2 22s ease-in-out infinite alternate;
}
@keyframes drift1 { from { transform: translate(0, 0) scale(1); } to { transform: translate(-160px, 140px) scale(1.25); } }
@keyframes drift2 { from { transform: translate(0, 0) scale(1); } to { transform: translate(180px, -120px) scale(1.2); } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
@keyframes shine { to { background-position: 200% center; } }
@keyframes floaty { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
@keyframes glow { 0%, 100% { box-shadow: 0 0 0 0 rgba(255, 217, 61, 0.55); } 50% { box-shadow: 0 0 0 9px rgba(255, 217, 61, 0); } }
@keyframes logofloat { 0%, 100% { transform: rotate(-4deg) translateY(0); } 50% { transform: rotate(-4deg) translateY(-3px); } }

.page-title, .page-sub, .hero-sub, .section-title, .empty, .docbar { animation: fadeUp 0.6s ease both; }
.hero-title {
    background: linear-gradient(90deg, #131B33, #2748E8, #7A5CFF, #2748E8, #131B33);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeUp 0.6s ease both, shine 7s linear infinite;
}
.steps { animation: fadeUp 0.7s ease 0.15s both; }
.step { animation: fadeUp 0.5s ease both; }
.step:nth-child(2) { animation-delay: 0.3s; }
.step:nth-child(3) { animation-delay: 0.45s; }
.step:nth-child(4) { animation-delay: 0.6s; }
.step:nth-child(5) { animation-delay: 0.75s; }
.cap { transition: background 0.2s; }
.cap:hover { background: #F6F8FF; }
.cap svg { animation: floaty 4s ease-in-out infinite; }
.cap:nth-child(2) svg { animation-delay: 0.5s; }
.cap:nth-child(3) svg { animation-delay: 1s; }
.cap:nth-child(4) svg { animation-delay: 1.5s; }
.brand-mark { animation: glow 3s ease-in-out infinite, logofloat 4s ease-in-out infinite; }
[data-testid="stChatMessage"] { animation: fadeUp 0.4s ease both; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
.stButton > button, .stDownloadButton > button { transition: background 0.15s, transform 0.15s; }

@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
"""

st.markdown(
    "<style>" + " ".join(l.strip() for l in CSS.splitlines()) + "</style>",
    unsafe_allow_html=True
)


# ==========================================
# SESSION STATE
# ==========================================

if "nav" not in st.session_state:
    st.session_state.nav = "Home"

if "rag" not in st.session_state:
    st.session_state.rag = StudyRAG()

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "doc_pages" not in st.session_state:
    st.session_state.doc_pages = 0

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = 0

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {
        "AI Tutor": [],
        "Ask Your PDF": []
    }

if "pending_q" not in st.session_state:
    st.session_state.pending_q = None

if "generated_result" not in st.session_state:
    st.session_state.generated_result = None

if "generated_filename" not in st.session_state:
    st.session_state.generated_filename = None

if "generated_title" not in st.session_state:
    st.session_state.generated_title = None

if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    render(f"""
    <div class="brand">
        <div class="brand-mark">{icon("logo")}</div>
        <div>
            <div class="brand-name">{APP_NAME}</div>
            <div class="brand-sub">Your study assistant</div>
        </div>
    </div>
    """)

    mode = st.radio(
        "Choose a workspace",
        NAV_PAGES,
        key="nav",
        label_visibility="collapsed"
    )

    render('<div class="side-label">Study material</div>')

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.caption("Text-based PDFs work best.")

    if uploaded_file is not None:

        if st.button("Process document", use_container_width=True):

            try:

                with st.spinner("Reading your study material..."):

                    upload_dir = Path("data/documents")
                    upload_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    # Use only the base file name for safety
                    pdf_path = upload_dir / Path(uploaded_file.name).name

                    pdf_path.write_bytes(
                        uploaded_file.getvalue()
                    )

                    pages = extract_pdf_text(str(pdf_path))

                    chunks = create_chunks(pages)

                    if not chunks:
                        raise ValueError(
                            "No readable text found in this PDF. "
                            "Try a text-based PDF instead of a scan."
                        )

                    total_chunks = st.session_state.rag.load_document(
                        chunks
                    )

                    st.session_state.document_loaded = True
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.doc_pages = len(pages)
                    st.session_state.doc_chunks = total_chunks

                    st.session_state.chat_histories[
                        "Ask Your PDF"
                    ] = []

                    st.session_state.generated_result = None

            except Exception as error:

                st.error(f"Processing failed: {error}")

    if st.session_state.document_loaded:

        render(f"""
        <div class="doc-card">
            <div class="doc-card-title">{html.escape(st.session_state.document_name)}</div>
            <div class="doc-card-meta">{st.session_state.doc_pages} pages, {st.session_state.doc_chunks} chunks</div>
            <div class="doc-card-state">Ready to use</div>
        </div>
        """)

    else:

        render("""
        <div class="doc-card">
            <div class="doc-card-title">No document yet</div>
            <div class="doc-card-meta">Choose a PDF, then press Process document.</div>
        </div>
        """)

    render('<div class="side-label">Tutor style</div>')

    tutor_style = st.selectbox(
        "Tutor style",
        list(TUTOR_STYLES.keys()),
        key="tutor_style",
        label_visibility="collapsed"
    )

    render(f"""
    <div class="engine">
        <div class="side-label" style="margin-top:0">This session</div>
        <div class="engine-row">Questions asked: {st.session_state.questions_asked}</div>
    </div>
    <div class="engine" style="margin-top:18px">
        <div class="side-label" style="margin-top:0">Powered by</div>
        <div class="engine-row">Gemini Flash</div>
        <div class="engine-row">FAISS search</div>
        <div class="engine-row">Sentence Transformers</div>
    </div>
    """)


# ==========================================
# HOME PAGE
# ==========================================

if mode == "Home":

    left, right = st.columns([1.25, 1], gap="large")

    with left:

        render(f"""
        <div class="hero-title">Ask your notes anything.</div>
        <div class="hero-sub">
            Upload a PDF and {APP_NAME} answers your questions from it,
            writes revision notes and quizzes you. Every answer shows
            the pages it came from. Stay on track with the built-in
            focus timer.
        </div>
        """)

        b1, b2, _ = st.columns([1, 1, 0.5])

        with b1:
            st.button(
                "Ask your PDF",
                key="cta_pdf",
                on_click=go_to,
                args=("Ask Your PDF",),
                use_container_width=True
            )

        with b2:
            st.button(
                "Open AI Tutor",
                key="cta_tutor",
                on_click=go_to,
                args=("AI Tutor",),
                use_container_width=True
            )

    with right:

        render(f"""
        <div class="steps">
            <div class="steps-title">How it works</div>
            <div class="step">
                <div class="step-n">1</div>
                <div>
                    <div class="step-t">Upload</div>
                    <div class="step-d">Choose a PDF in the sidebar.</div>
                </div>
            </div>
            <div class="step">
                <div class="step-n">2</div>
                <div>
                    <div class="step-t">Process</div>
                    <div class="step-d">{APP_NAME} reads the text and indexes it for search.</div>
                </div>
            </div>
            <div class="step">
                <div class="step-n">3</div>
                <div>
                    <div class="step-t">Ask or generate</div>
                    <div class="step-d">Chat with your notes, or create a summary, notes or quiz.</div>
                </div>
            </div>
            <div class="step">
                <div class="step-n">4</div>
                <div>
                    <div class="step-t">Check the source</div>
                    <div class="step-d">Each answer lists the pages it used.</div>
                </div>
            </div>
        </div>
        """)

    render('<div class="section-title">What you can do</div>')

    render(f"""
    <div class="cap-grid">
        <div class="cap">
            {icon("chat")}
            <div class="cap-t">AI Tutor</div>
            <div class="cap-d">Ask any question, and pick a style: simple, exam-focused or step-by-step.</div>
        </div>
        <div class="cap">
            {icon("file")}
            <div class="cap-t">Ask Your PDF</div>
            <div class="cap-d">Get answers taken from your own notes, with source pages.</div>
        </div>
        <div class="cap">
            {icon("notes")}
            <div class="cap-t">Summaries and notes</div>
            <div class="cap-d">Turn a chapter into key ideas and revision notes.</div>
        </div>
        <div class="cap">
            {icon("quiz")}
            <div class="cap-t">Quizzes</div>
            <div class="cap-d">Practise with questions based on your own material.</div>
        </div>
    </div>
    """)

    render('<div class="section-title">How it is built</div>')

    render("""
    <div class="spec-grid">
        <div><div class="spec-k">Language model</div><div class="spec-v">Gemini Flash</div></div>
        <div><div class="spec-k">Document search</div><div class="spec-v">FAISS</div></div>
        <div><div class="spec-k">Embeddings</div><div class="spec-v">Sentence Transformers</div></div>
        <div><div class="spec-k">Privacy</div><div class="spec-v">Relevant excerpts are sent to Google's Gemini API</div></div>
    </div>
    """)


# ==========================================
# AI TUTOR
# ==========================================

elif mode == "AI Tutor":

    page_header(
        "AI Tutor",
        "Ask about any topic and get a clear explanation, "
        "with examples when they help. Change the tutor style "
        "in the sidebar."
    )

    history = st.session_state.chat_histories["AI Tutor"]

    question = st.chat_input("Ask your AI tutor anything...")

    if not question and st.session_state.pending_q:
        question = st.session_state.pending_q
        st.session_state.pending_q = None

    if not history and not question:

        render('<div class="section-title" style="margin-top:8px">Try one of these</div>')

        suggestion_buttons(TUTOR_PROMPTS, "prompt")

    for message in history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question:

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    styled_question = (
                        TUTOR_STYLES[tutor_style] + question
                    )

                    answer = ask_tutor(
                        styled_question,
                        history=history
                    )

                    st.markdown(answer)

                    history.append({
                        "role": "user",
                        "content": question
                    })

                    history.append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.session_state.questions_asked += 1

                except Exception as error:

                    st.error(f"Something went wrong: {error}")

    chat_toolbar("tutor", "AI Tutor", history)


# ==========================================
# ASK YOUR PDF
# ==========================================

elif mode == "Ask Your PDF":

    page_header(
        "Ask Your PDF",
        "Ask questions about your document. Answers come from your "
        "own material, and the source passages are shown with each one."
    )

    if not st.session_state.document_loaded:

        empty_state(
            "No document loaded",
            "Upload a PDF in the sidebar, then press Process document "
            "to start asking questions."
        )

    else:

        doc_bar()

        history = st.session_state.chat_histories["Ask Your PDF"]

        question = st.chat_input(
            "Ask something from your uploaded PDF..."
        )

        if not question and st.session_state.pending_q:
            question = st.session_state.pending_q
            st.session_state.pending_q = None

        if not history and not question:

            render('<div class="section-title" style="margin-top:8px">Try one of these</div>')

            suggestion_buttons(PDF_PROMPTS, "prompt_pdf")

        for message in history:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question:

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner("Searching your notes..."):

                    try:

                        answer, sources = (
                            st.session_state.rag.ask_document(
                                question,
                                history=history
                            )
                        )

                        st.markdown(answer)

                        with st.expander("View source passages"):

                            for source in sources:

                                page = html.escape(str(source["page"]))
                                text = html.escape(
                                    " ".join(str(source["text"]).split())
                                )

                                render(f"""
                                <div class="source">
                                    <div class="source-head">
                                        <span class="source-page">Page {page}</span>
                                        <span class="source-score">Similarity {source["score"]:.3f}</span>
                                    </div>
                                    <div class="source-text"><mark>{text}</mark></div>
                                </div>
                                """)

                        history.append({
                            "role": "user",
                            "content": question
                        })

                        history.append({
                            "role": "assistant",
                            "content": answer
                        })

                        st.session_state.questions_asked += 1

                    except Exception as error:

                        st.error(f"Something went wrong: {error}")

        chat_toolbar("pdf", "Ask Your PDF", history)


# ==========================================
# STUDY TOOLS
# ==========================================

elif mode == "Study Tools":

    page_header(
        "Study Tools",
        "Turn your document into a summary, revision notes or a "
        "practice quiz, then download the result."
    )

    if not st.session_state.document_loaded:

        empty_state(
            "No document loaded",
            "Upload a PDF in the sidebar, then press Process document "
            "to unlock these tools."
        )

    else:

        doc_bar()

        def run_tool(label, func, filename):

            try:

                chunks = st.session_state.rag.chunks

                with st.spinner(f"Creating your {label.lower()}..."):
                    result = func(chunks)

                st.session_state.generated_result = result
                st.session_state.generated_filename = filename
                st.session_state.generated_title = label

            except Exception as error:

                st.error(f"Generation failed: {error}")

        tab_summary, tab_notes, tab_quiz = st.tabs(
            ["Summary", "Revision notes", "Quiz"]
        )

        with tab_summary:

            render("""
            <div class="tool-desc">
                The important ideas and concepts from your document,
                in a short and readable format.
            </div>
            """)

            if st.button("Generate summary", key="gen_summary"):
                run_tool("Summary", generate_summary, "study_summary.txt")

        with tab_notes:

            render("""
            <div class="tool-desc">
                Structured notes with headings and bullet points,
                ready for revision.
            </div>
            """)

            if st.button("Generate revision notes", key="gen_notes"):
                run_tool("Revision notes", generate_notes, "revision_notes.txt")

        with tab_quiz:

            render("""
            <div class="tool-desc">
                Multiple-choice questions with answers and short
                explanations, based on your material.
            </div>
            """)

            if st.button("Generate quiz", key="gen_quiz"):
                run_tool("Quiz", generate_quiz, "practice_quiz.txt")

        if st.session_state.generated_result:

            render(f"""
            <div class="section-title" style="margin-top:32px">
                {html.escape(st.session_state.generated_title or "Result")}
            </div>
            """)

            words = len(str(st.session_state.generated_result).split())
            st.caption(f"About {words} words")

            with st.container(border=True):
                st.markdown(st.session_state.generated_result)

            st.download_button(
                "Download as text file",
                data=st.session_state.generated_result,
                file_name=st.session_state.generated_filename,
                mime="text/plain"
            )


# ==========================================
# FOCUS TIMER
# ==========================================

elif mode == "Focus Timer":

    page_header(
        "Focus Timer",
        "Study in short, focused rounds with breaks in between, "
        "and keep a scratchpad for quick notes."
    )

    t1, t2, t3 = st.columns(3)

    with t1:
        focus_min = st.number_input(
            "Focus (minutes)", min_value=1, max_value=120, value=25
        )
    with t2:
        short_min = st.number_input(
            "Short break (minutes)", min_value=1, max_value=30, value=5
        )
    with t3:
        long_min = st.number_input(
            "Long break (minutes)", min_value=1, max_value=60, value=15
        )

    components.html(
        TIMER_HTML
        .replace("__FOCUS__", str(int(focus_min)))
        .replace("__SHORT__", str(int(short_min)))
        .replace("__LONG__", str(int(long_min))),
        height=540
    )

    render('<div class="section-title" style="margin-top:28px">Scratchpad</div>')

    scratch = st.text_area(
        "Scratchpad",
        key="scratchpad",
        height=180,
        placeholder="Jot down questions, formulas or things to revisit...",
        label_visibility="collapsed"
    )

    if scratch.strip():
        st.download_button(
            "Download scratchpad",
            data=scratch,
            file_name="scratchpad.txt",
            mime="text/plain"
        )