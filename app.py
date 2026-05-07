import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.translations import t
from utils.crop_utils import predict_crop, predict_top_crops, get_crop_info
from utils.fertilizer_utils import get_all_crops, get_soil_types_for_crop, recommend_fertilizer
from utils.mandi_utils import get_states, get_commodities, get_prices

st.set_page_config(page_title="Krushi Mitra", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(160deg, #1a4731 0%, #2d6a4f 100%); }
    [data-testid="stSidebar"] * { color: #fefae0 !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * { color: #1a4731 !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] { background: white !important; border-radius: 8px !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #fefae0 !important; }
    .main-header { background: linear-gradient(135deg, #1a4731 0%, #2d6a4f 60%, #52b788 100%); padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem; }
    .main-header h1 { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 700; margin: 0; color: #e9c46a; }
    .main-header p { font-size: 1.05rem; color: #c8e6c9; margin: 0.4rem 0 0 0; }
    .feature-card { background: white; border: 1px solid #e8f5e9; border-radius: 14px; padding: 1.5rem; text-align: center; box-shadow: 0 2px 12px rgba(26,71,49,0.07); }
    .feature-card .icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
    .feature-card h3 { color: #1a4731; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
    .feature-card p { color: #555; font-size: 0.9rem; line-height: 1.5; }
    .result-box { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 2px solid #52b788; border-radius: 14px; padding: 1.5rem 2rem; margin-top: 1.5rem; }
    .result-box h2 { color: #1a4731; font-family: 'Playfair Display', serif; font-size: 1.8rem; margin: 0 0 0.3rem 0; text-transform: capitalize; }
    .result-box .confidence { color: #2d6a4f; font-weight: 600; font-size: 0.95rem; }
    .result-box .crop-detail { background: white; border-radius: 8px; padding: 0.8rem 1rem; margin-top: 0.8rem; font-size: 0.9rem; color: #444; }
    .advice-item { background: #fffbeb; border-left: 4px solid #e9c46a; border-radius: 0 8px 8px 0; padding: 0.7rem 1rem; margin: 0.5rem 0; font-size: 0.93rem; color: #333; }
    .advice-good { background: #f0fdf4; border-left: 4px solid #52b788; }
    .section-title { font-family: 'Playfair Display', serif; font-size: 1.7rem; color: #1a4731; font-weight: 700; margin-bottom: 1rem; padding-bottom: 0.4rem; border-bottom: 3px solid #52b788; }
    .stButton > button { background: linear-gradient(135deg, #2d6a4f, #52b788); color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem; width: 100%; }
    div[data-testid="stMetric"] { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 0.8rem 1rem; }
    .fert-input-card { background: white; border: 1.5px solid #bbf7d0; border-radius: 16px; padding: 2rem 2.2rem 1.5rem 2.2rem; box-shadow: 0 2px 12px rgba(26,71,49,0.07); margin-bottom: 1.5rem; }
    .fert-result-header { background: linear-gradient(135deg, #1a4731 0%, #2d6a4f 60%, #52b788 100%); border-radius: 14px; padding: 1.4rem 2rem; margin: 1.2rem 0; color: white; }
    .fert-result-header h2 { color: #e9c46a; font-family: 'Playfair Display', serif; margin: 0 0 0.2rem 0; font-size: 1.7rem; }
    .fert-result-header p  { color: #c8e6c9; margin: 0; font-size: 0.95rem; }
    .fert-dose-box { background: #fffbeb; border: 2px solid #e9c46a; border-radius: 12px; padding: 1.2rem 1.5rem; margin: 1rem 0; }
    .fert-dose-box h4 { color: #92400e; margin: 0 0 0.5rem 0; font-size: 1rem; }
    .fert-dose-box p  { color: #333; margin: 0; font-size: 0.95rem; font-weight: 500; }
    .npk-pill { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 99px; font-size: 0.85rem; font-weight: 700; margin: 0 0.3rem 0.3rem 0; }
    .npk-n { background: #dcfce7; color: #166534; }
    .npk-p { background: #dbeafe; color: #1e3a8a; }
    .npk-k { background: #fef9c3; color: #713f12; }
    .npk-ph { background: #fce7f3; color: #831843; }
    .npk-w  { background: #e0f2fe; color: #0c4a6e; }
</style>
""", unsafe_allow_html=True)

if "lang" not in st.session_state:
    st.session_state.lang = "English"

with st.sidebar:
    st.markdown("## 🌾 Krushi Mitra")
    st.markdown("---")
    lang = st.selectbox(t("select_language", st.session_state.lang), ["English", "Hindi", "Odia"],
                        index=["English", "Hindi", "Odia"].index(st.session_state.lang))
    if lang != st.session_state.lang:
        st.session_state.lang = lang
        st.session_state.pop("_active_page", None)
        st.rerun()
    L = lang
    st.markdown("---")
    nav_options = [t("nav_home",L), t("nav_crop",L), t("nav_fertilizer",L), t("nav_disease",L), t("nav_mandi",L)]
    default_page = st.session_state.get("_active_page", nav_options[0])
    if default_page not in nav_options:
        default_page = nav_options[0]
    page = st.radio("Navigate", nav_options, index=nav_options.index(default_page))
    st.session_state["_active_page"] = page
    st.markdown("---")
    st.markdown("<small style='color:#a5d6a7'>Krushi Mitra v1.0<br>BCA Project · SOA University</small>", unsafe_allow_html=True)

L = st.session_state.lang

# HOME
if page == t("nav_home", L):
    st.markdown(f'<div class="main-header"><h1>🌾 {t("app_title",L)}</h1><p>{t("app_subtitle",L)}</p></div>', unsafe_allow_html=True)
    st.markdown(f"### {t('welcome', L)}")
    st.markdown(t('home_desc', L))
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,icon,feat,desc in [(c1,"🌱",t("feature_crop",L),t("feature_crop_desc",L)),(c2,"🧪",t("feature_fertilizer",L),t("feature_fertilizer_desc",L)),(c3,"🔬",t("feature_disease",L),t("feature_disease_desc",L)),(c4,"📊",t("feature_mandi",L),t("feature_mandi_desc",L))]:
        with col:
            st.markdown(f'<div class="feature-card"><div class="icon">{icon}</div><h3>{feat}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("📌 **Note:** This system uses AI models trained on agricultural datasets. Results are advisory only. Always consult your local agricultural extension officer for critical decisions.")

# CROP
elif page == t("nav_crop", L):
    from utils.crop_utils import predict_top_crops

    st.markdown(f'<div class="section-title">🌱 {t("crop_title",L)}</div>', unsafe_allow_html=True)
    st.markdown("#### Upload a photo of your soil to get started")
    st.caption("No technical knowledge needed — we'll guide you step by step.")

    # ── Progress bar styling ──────────────────────────────────────────────────
    st.markdown("""
    <style>
    .step-indicator { display:flex; gap:8px; margin: 1rem 0 1.5rem 0; }
    .step-dot { width:32px; height:32px; border-radius:50%; display:flex; align-items:center;
                justify-content:center; font-size:0.8rem; font-weight:700; }
    .step-dot.done  { background:#2d6a4f; color:white; }
    .step-dot.active{ background:#52b788; color:white; box-shadow:0 0 0 3px #bbf7d0; }
    .step-dot.todo  { background:#e8f5e9; color:#aaa; border:2px solid #c8e6c9; }
    .step-line { flex:1; height:2px; background:#e8f5e9; margin:auto; }
    .step-line.done { background:#2d6a4f; }
    .q-card { background:white; border:1.5px solid #bbf7d0; border-radius:14px;
               padding:1.5rem 1.8rem; margin:1rem 0; box-shadow:0 2px 10px rgba(26,71,49,0.06); }
    .q-title { color:#1a4731; font-size:1.05rem; font-weight:600; margin-bottom:1rem; }
    .slider-label { display:flex; justify-content:space-between;
                    font-size:0.8rem; color:#666; margin-top:4px; }
    .crop-rank-card { background:white; border-radius:12px; padding:1.2rem 1.5rem;
                      margin:0.6rem 0; border-left:5px solid #52b788;
                      box-shadow:0 1px 6px rgba(26,71,49,0.08); }
    .crop-rank-card.secondary { border-left-color:#e9c46a; opacity:0.92; }
    .crop-rank-card.tertiary  { border-left-color:#adb5bd; opacity:0.85; }
    .crop-rank-num { font-size:0.75rem; font-weight:700; color:#888; text-transform:uppercase;
                     letter-spacing:0.05em; margin-bottom:4px; }
    .crop-rank-name { font-size:1.3rem; font-weight:700; color:#1a4731;
                      font-family:'Playfair Display',serif; text-transform:capitalize; }
    .conf-bar-wrap { background:#e8f5e9; border-radius:99px; height:8px; margin:8px 0 4px 0; overflow:hidden; }
    .conf-bar { height:8px; border-radius:99px; background:linear-gradient(90deg,#2d6a4f,#52b788); }
    .conf-bar.yellow { background:linear-gradient(90deg,#d4a017,#e9c46a); }
    .conf-bar.grey   { background:linear-gradient(90deg,#6c757d,#adb5bd); }
    </style>
    """, unsafe_allow_html=True)

    # ── Initialise session state for step-by-step wizard ─────────────────────
    for key, default in [("crop_step", 0), ("cq1", None), ("cq2", None), ("cq3", None),
                         ("crop_result", None), ("soil_detected", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Step 0: Upload soil photo ─────────────────────────────────────────────
    soil_image = st.file_uploader("📷 Upload Soil Photo", type=["jpg","jpeg","png"], key="soil_img")

    if not soil_image:
        st.info("👆 Upload a clear photo of your soil to begin. Dry or slightly moist soil works best.")
        # Reset wizard if they remove the image
        st.session_state.crop_step = 0
        st.session_state.crop_result = None
    else:
        from utils.soil_utils import detect_soil_type, SOIL_TYPES, estimate_npk

        # Detect soil only once per image
        img_bytes = soil_image.read()
        if st.session_state.soil_detected is None or st.session_state.get("_last_img") != soil_image.name:
            detected_soil, soil_confidence, avg_rgb, detection_methof = detect_soil_type(img_bytes)

            if soil_confidence < 50.0:
                st.error(" Couldn't detect a valid soil type. . .")
                st.stop()
            st.session_state.soil_detected = (detected_soil, soil_confidence)
            st.session_state["_last_img"] = soil_image.name
            st.session_state.crop_step = 1
            st.session_state.cq1 = None
            st.session_state.cq2 = None
            st.session_state.cq3 = None
            st.session_state.crop_result = None

        detected_soil, soil_confidence = st.session_state.soil_detected
        info = SOIL_TYPES[detected_soil]

        # ── Soil result card ──────────────────────────────────────────────────
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(soil_image, caption="Your Soil Sample", use_container_width=True)
        with col_info:
            st.markdown(f"""
            <div class="result-box" style="margin-top:0">
                <h2 style="font-size:1.3rem">🪨 Detected: {detected_soil}</h2>
                <div class="confidence">Confidence: {soil_confidence}%</div>
                <div class="crop-detail">
                    {info['description']}<br><br>
                    <strong>Characteristics:</strong> {info['characteristics']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Step indicator ────────────────────────────────────────────────────
        step = st.session_state.crop_step
        display_step = int(step)   # 3.5 → 3 for indicator purposes
        def dot(n):
            if n < display_step:   return f'<div class="step-dot done">✓</div>'
            if n == display_step:  return f'<div class="step-dot active">{n}</div>'
            return                     f'<div class="step-dot todo">{n}</div>'
        def line(n):
            cls = "done" if n < display_step else ""
            return f'<div class="step-line {cls}"></div>'

        st.markdown(f"""
        <div class="step-indicator">
            {dot(1)}{line(1)}{dot(2)}{line(2)}{dot(3)}
        </div>
        """, unsafe_allow_html=True)

        # ── MCQ button CSS ────────────────────────────────────────────────────
        st.markdown("""
        <style>
        .mcq-grid { display:flex; flex-wrap:wrap; gap:10px; margin:0.8rem 0 1.2rem 0; }
        div[data-testid="stButton"].mcq-btn > button {
            background: white;
            color: #1a4731;
            border: 2px solid #bbf7d0;
            border-radius: 12px;
            padding: 0.7rem 1.2rem;
            font-size: 0.95rem;
            font-weight: 500;
            width: auto;
            text-align: left;
            box-shadow: 0 1px 4px rgba(26,71,49,0.07);
            transition: all 0.15s;
        }
        div[data-testid="stButton"].mcq-btn > button:hover {
            border-color: #52b788;
            background: #f0fdf4;
            color: #1a4731;
        }
        </style>
        """, unsafe_allow_html=True)

        # ── Question 1: Drainage (slider — step 1 only) ───────────────────────
        if step == 1:
            q1_opts = [
                "Drains very fast (within minutes)",
                "Drains normally (few hours)",
                "Stays wet for a long time",
            ]
            q1_labels = ["Very Fast", "Normal", "Stays Wet"]
            st.markdown('<div class="q-card">', unsafe_allow_html=True)
            st.markdown('<div class="q-title">💧 Question 1 of 3 &nbsp;·&nbsp; After rain or irrigation, how does water behave in your field?</div>', unsafe_allow_html=True)
            q1_idx = st.select_slider("", options=[0, 1, 2],
                                      format_func=lambda x: q1_labels[x], key="q1_slider")
            st.markdown('<div class="slider-label"><span>Drains very fast</span><span>Stays wet long</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Next →", key="next1"):
                st.session_state.cq1 = q1_opts[q1_idx]
                st.session_state.crop_step = 2
                st.rerun()

        elif step >= 2:
            st.caption(f"✅ Q1 answer: **{st.session_state.cq1}**")

        # ── Question 2: Rainfall (slider — step 2 only) ───────────────────────
        if step == 2:
            q2_opts = [
                "Low (dry region, <600mm/year)",
                "Moderate (600-1200mm/year)",
                "High (>1200mm/year, like Odisha coast)",
            ]
            q2_labels = ["Low", "Moderate", "High"]
            st.markdown('<div class="q-card">', unsafe_allow_html=True)
            st.markdown('<div class="q-title">🌧️ Question 2 of 3 &nbsp;·&nbsp; What is the annual rainfall like in your region?</div>', unsafe_allow_html=True)
            q2_idx = st.select_slider("", options=[0, 1, 2],
                                      format_func=lambda x: q2_labels[x], key="q2_slider")
            st.markdown('<div class="slider-label"><span>Low (&lt;600mm)</span><span>High (&gt;1200mm)</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            col_back, col_next = st.columns([1, 4])
            with col_back:
                if st.button("← Back", key="back2"):
                    st.session_state.crop_step = 1
                    st.rerun()
            with col_next:
                if st.button("Next →", key="next2"):
                    st.session_state.cq2 = q2_opts[q2_idx]
                    st.session_state.crop_step = 3
                    st.rerun()

        elif step >= 3:
            st.caption(f"✅ Q2 answer: **{st.session_state.cq2}**")

        # ── Question 3: Previous crop (MCQ buttons — step 3 only) ─────────────
        if step == 3:
            q3_options = [
                ("Legume",    "Legume (dal, soybean, groundnut)"),
                ("Cereal",    "Cereal (rice, wheat, maize)"),
                ("Vegetable", "Vegetable"),
                ("Cash Crop", "Cash crop (cotton, sugarcane)"),
                ("Not Sure",  "This is my first crop / Not sure"),
            ]

            st.markdown('<div class="q-card">', unsafe_allow_html=True)
            st.markdown('<div class="q-title">🌾 Question 3 of 3 &nbsp;·&nbsp; What was the last crop grown on this land?</div>', unsafe_allow_html=True)
            st.markdown("<p style='color:#555;font-size:0.9rem;margin:0 0 1rem 0'>Select the option that best describes your previous crop:</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            cols = st.columns(5)
            for i, (label, full_val) in enumerate(q3_options):
                with cols[i]:
                    if st.button(label, key=f"q3_btn_{i}", use_container_width=True):
                        st.session_state.cq3 = full_val
                        st.session_state.crop_step = 3.5
                        st.rerun()

            col_back, _ = st.columns([1, 4])
            with col_back:
                if st.button("← Back", key="back3"):
                    st.session_state.crop_step = 2
                    st.rerun()

        # ── Submit: triggered right after Q3 MCQ selection ────────────────────
        if st.session_state.crop_step == 3.5 and st.session_state.cq3:
            st.caption(f"✅ Q3 answer: **{st.session_state.cq3}**")
            with st.spinner("Analyzing your soil and field conditions..."):
                try:
                    params = estimate_npk(
                        detected_soil,
                        st.session_state.cq1,
                        st.session_state.cq2,
                        st.session_state.cq3,
                    )
                    top_crops = predict_top_crops(
                        params["N"], params["P"], params["K"],
                        params["temperature"], params["humidity"],
                        params["ph"], params["rainfall"],
                        top_n=3, threshold=8.0,
                    )
                    st.session_state.crop_result = (top_crops, params)
                    st.session_state.crop_step = 4
                    st.rerun()
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")

        # ── Results ───────────────────────────────────────────────────────────
        if display_step >= 4 and st.session_state.crop_result:
            top_crops, params = st.session_state.crop_result
            rank_styles = [
                ("🥇 Best Match",   "crop-rank-card",           "conf-bar"),
                ("🥈 Alternative",  "crop-rank-card secondary",  "conf-bar yellow"),
                ("🥉 Also Suitable","crop-rank-card tertiary",   "conf-bar grey"),
            ]
            st.markdown("---")
            st.markdown("### 🌾 Recommended Crops")
            if len(top_crops) == 1:
                st.caption("The model is very confident about a single crop for your conditions.")
            else:
                st.caption(f"Found **{len(top_crops)} suitable crops** for your soil and field conditions.")

            for i, (crop, conf) in enumerate(top_crops):
                label, card_cls, bar_cls = rank_styles[i] if i < 3 else rank_styles[2]
                st.markdown(f"""
                <div class="{card_cls}">
                    <div class="crop-rank-num">{label}</div>
                    <div class="crop-rank-name">{crop.title()}</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("🔍 View estimated soil parameters used"):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("N (est.)", params["N"])
                c2.metric("P (est.)", params["P"])
                c3.metric("K (est.)", params["K"])
                c4.metric("pH (est.)", params["ph"])
                st.caption(f"Estimated from: {detected_soil} + your answers · Rainfall: {params['rainfall']}mm · Temp: {params['temperature']}°C")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Start Over", key="restart"):
                for key in ["crop_step","cq1","cq2","cq3","crop_result","soil_detected","_last_img"]:
                    st.session_state.pop(key, None)
                st.rerun()

# FERTILIZER
elif page == t("nav_fertilizer", L):
    st.markdown(f'<div class="section-title">🧪 {t("fertilizer_title",L)}</div>', unsafe_allow_html=True)

    crops = get_all_crops()

    st.markdown('<div class="fert-input-card">', unsafe_allow_html=True)
    st.markdown("#### 🌾 Select your crop and soil type")
    st.caption("That's all we need — we'll handle the rest.")

    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("🌱 Crop", crops, help="Choose the crop you want to grow")
    with col2:
        soil_options = get_soil_types_for_crop(crop)
        soil_type = st.selectbox("🪨 Soil Type", soil_options, help="Select your field's soil type")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Get Fertilizer Recommendation", use_container_width=True):
        result = recommend_fertilizer(crop, soil_type)

        if result:
            # Header
            st.markdown(f"""
            <div class="fert-result-header">
                <h2>🌿 {result['crop']} on {result['soil_type']} Soil</h2>
                <p>📅 Season: {result['season']} &nbsp;|&nbsp; 🪨 Soil: {result['soil_type']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Recommended fertilizer
            st.markdown(f"""
            <div class="fert-dose-box">
                <h4>💊 Recommended Fertilizer</h4>
                <p>{result['fertilizer_name']}</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="fert-dose-box" style="border-color:#86efac">
                    <h4 style="color:#166534">📦 Dose</h4>
                    <p>{result['fertilizer_dose']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="fert-dose-box" style="border-color:#93c5fd">
                    <h4 style="color:#1e40af">🔧 Application Method</h4>
                    <p>{result['application_method']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Ideal NPK badges
            npk = result['ideal_npk']
            st.markdown("**📊 Ideal Soil Parameters for this combination:**")
            st.markdown(f"""
            <span class="npk-pill npk-n">N: {npk['N']} kg/ha</span>
            <span class="npk-pill npk-p">P: {npk['P']} kg/ha</span>
            <span class="npk-pill npk-k">K: {npk['K']} kg/ha</span>
            <span class="npk-pill npk-ph">pH: {npk['pH']}</span>
            <span class="npk-pill npk-w">Moisture: {npk['soil_moisture']}%</span>
            """, unsafe_allow_html=True)


        else:
            st.warning("⚠️ No fertilizer data found for this crop and soil combination. Try a different soil type.")

# DISEASE
elif page == t("nav_disease", L):
    st.markdown(f'<div class="section-title">🔬 {t("disease_title",L)}</div>', unsafe_allow_html=True)
    st.info("📸 Upload a clear image of the affected leaf. Supported crops: Apple, Corn, Grape, Tomato, Potato, Peach, Pepper, Strawberry and more.")

    uploaded = st.file_uploader(t("upload_image",L), type=["jpg","jpeg","png"])

    if uploaded:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded, caption="Uploaded Leaf Image", use_container_width=True)

        with col2:
            if st.button(t("detect_disease", L)):
                with st.spinner(t("loading", L)):
                    try:
                        from utils.disease_utils import predict_disease, SEVERITY_COLOR
                        img_bytes = uploaded.read()
                        class_key, display_name, confidence, info = predict_disease(img_bytes)

                        severity = info.get('severity', 'Unknown')
                        sev_color = SEVERITY_COLOR.get(severity, '#555')
                        is_healthy = severity == 'None'
                        box_icon = "✅" if is_healthy else "⚠️"

                        st.markdown(f"""
                        <div class="result-box">
                            <h2>{box_icon} {display_name}</h2>
                            <div class="confidence">Confidence: {confidence}%</div>
                            <div class="crop-detail">
                                <span style="background:{sev_color};color:white;padding:2px 10px;border-radius:12px;font-size:0.82rem;font-weight:600">{severity} Severity</span>
                                <br><br>
                                🦠 <strong>Cause:</strong> {info.get('cause','N/A')}<br><br>
                                🔍 <strong>Symptoms:</strong> {info.get('symptoms','N/A')}<br><br>
                                🛡️ <strong>Prevention/Treatment:</strong> {info.get('prevention','N/A')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    except ImportError:
                        st.warning("⚙️ PyTorch is not installed. Run: `pip install torch torchvision` to enable this module.")
                    except Exception as e:
                        st.error(f"Detection failed: {str(e)}")

# MANDI
elif page == t("nav_mandi", L):
    st.markdown(f'<div class="section-title">📊 {t("mandi_title",L)}</div>', unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        states = ["All States"] + get_states()
        sel_state = st.selectbox(t("select_state",L), states)
        sf = None if sel_state == "All States" else sel_state
    with col2:
        commodities = ["All Commodities"] + get_commodities(sf)
        sel_commodity = st.selectbox(t("select_commodity",L), commodities)
        cf = None if sel_commodity == "All Commodities" else sel_commodity
    df = get_prices(sf, cf)
    if df.empty:
        st.warning("No data for selected filters.")
    else:
        m1,m2,m3 = st.columns(3)
        m1.metric("Markets Listed", len(df))
        m2.metric("Avg Modal Price (₹/Qtl)", int(df["Modal Price (₹/Qtl)"].mean()))
        m3.metric("Commodities", df["Commodity"].nunique())
        st.markdown(f"<br>**{t('price_table',L)}**", unsafe_allow_html=True)
        st.dataframe(df.sort_values("Modal Price (₹/Qtl)", ascending=False), use_container_width=True, hide_index=True)
        st.caption("📌 Prices are simulated based on Agmarknet dataset structure. For real-time data, visit agmarknet.gov.in")
