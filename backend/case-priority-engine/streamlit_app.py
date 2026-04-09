import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Case Priority Tagger", page_icon="⚖️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.tag { display:inline-block; padding:3px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.tag-Critical { background:#fee2e2; color:#991b1b; }
.tag-Medium   { background:#fef9c3; color:#854d0e; }
.tag-Routine  { background:#dcfce7; color:#166534; }
.meta    { font-size:13px; color:#6b7280; margin-top:4px; }
.summary { font-size:14px; color:#374151; margin-top:8px; line-height:1.6; }
.score   { font-size:32px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Check backend is alive ────────────────────────────────────
try:
    requests.get(f"{BACKEND_URL}/health", timeout=3)
except Exception:
    st.error("Backend is not running. Start it first — see README for instructions.")
    st.stop()

# ── Session state ─────────────────────────────────────────────
if "cases" not in st.session_state:
    st.session_state.cases = []

# ── Header ────────────────────────────────────────────────────
st.markdown("## ⚖️ Case Priority Tagger")
st.markdown("Upload pending case PDFs — AI ranks them by urgency.")
st.markdown("---")

# ── Search ────────────────────────────────────────────────────
search = st.text_input(
    "Search cases",
    placeholder="e.g. murder, kidnapping, Lahore High Court..."
)

# ── Upload ────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload case PDFs", type=["pdf"], accept_multiple_files=True,
    help="Upload up to 10 PDF files. Each file = one case."
)

col1, col2 = st.columns([3, 1])
with col1:
    process = st.button(
        "Process & Rank", type="primary",
        use_container_width=True,
        disabled=not uploaded_files
    )
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.cases = []
        st.rerun()

# ── Send to FastAPI ───────────────────────────────────────────
if process and uploaded_files:
    with st.spinner("Processing cases..."):
        files_payload = [
            ("files", (f.name, f.read(), "application/pdf"))
            for f in uploaded_files
        ]
        try:
            response = requests.post(
                f"{BACKEND_URL}/process",
                files=files_payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.cases = data.get("cases", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.rerun()

# ── Results ───────────────────────────────────────────────────
cases = st.session_state.cases

if not cases:
    st.markdown(
        "<br><p style='text-align:center;color:#9ca3af'>No cases yet — upload PDFs above</p>",
        unsafe_allow_html=True
    )
else:
    filtered = cases
    if search.strip():
        q = search.strip().lower()
        filtered = [c for c in cases if
                    q in c["title"].lower() or
                    q in c["filename"].lower() or
                    q in c["signals"].get("case_type", "").lower() or
                    q in c["signals"].get("court",     "").lower() or
                    q in c["signals"].get("summary",   "").lower()]

    st.markdown(f"**{len(filtered)} case(s)**")

    for rank, case in enumerate(filtered, 1):
        sig   = case["signals"]
        score = case["score"]
        tag   = case["tag"]
        color = {"Critical": "#dc2626", "Medium": "#d97706", "Routine": "#16a34a"}[tag]

        flags = []
        if sig.get("accused_in_custody"): flags.append("🔒 In custody")
        if sig.get("involves_minor"):     flags.append("👦 Minor")
        if sig.get("involves_woman"):     flags.append("👩 Woman")
        if sig.get("involves_elder"):     flags.append("👴 Elder")

        with st.expander(f"#{rank}  {case['title']}", expanded=(rank <= 3)):
            col_a, col_b = st.columns([3, 1])

            with col_a:
                st.markdown(f'<span class="tag tag-{tag}">{tag}</span>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="meta">'
                    f'{sig.get("court","—")} &nbsp;·&nbsp; '
                    f'{sig.get("case_type","—").capitalize()} &nbsp;·&nbsp; '
                    f'{sig.get("section","—")}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                if flags:
                    st.markdown(" &nbsp; ".join(flags), unsafe_allow_html=True)
                st.markdown(
                    f'<div class="summary">{sig.get("summary","—")}</div>',
                    unsafe_allow_html=True
                )

            with col_b:
                st.markdown(
                    f'<div style="text-align:center;padding-top:8px">'
                    f'<span class="score" style="color:{color}">{score}</span>'
                    f'<br><span style="font-size:12px;color:#6b7280">/ 100</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
