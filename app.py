import streamlit as st
import base64
import os

st.set_page_config(page_title="AtmoSound", layout="wide", initial_sidebar_state="expanded")

if "page" not in st.session_state:
    st.session_state.page = "home"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif; box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background-color: #0d0d0d; }

/* FORCE SIDEBAR ALWAYS VISIBLE */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] {
    transform: none !important;
    min-width: 244px !important;
    max-width: 244px !important;
    width: 244px !important;
    visibility: visible !important;
    display: block !important;
    position: relative !important;
    background-color: #1a1a2e !important;
    padding-top: 20px;
}
button[data-testid="baseButton-header"] { display: none !important; }
[data-testid="stSidebar"] button[kind="header"] { display: none !important; }

/* Sidebar text white — but NOT the logo button */
[data-testid="stSidebar"] .sidebar-section-label,
[data-testid="stSidebar"] .sidebar-item,
[data-testid="stSidebar"] span { color: white !important; }

/* All sidebar buttons default style */
[data-testid="stSidebar"] .stButton > button {
    background-color: transparent !important;
    color: white !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 7px 20px !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    width: 100% !important;
    text-align: left !important;
    height: auto !important;
    position: relative !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #ff2d78 !important;
    border-radius: 20px !important;
}

/* LOGO BUTTON — gradient text, overrides the white rule */
[data-testid="stSidebar"] .stButton:first-child > button {
    font-size: 24px !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #ff2d78, #a855f7) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    color: transparent !important;
    padding: 0 20px 20px 20px !important;
    border-radius: 0 !important;
    width: 100% !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton:first-child > button:hover {
    background: linear-gradient(90deg, #ff2d78, #a855f7) !important;
    -webkit-background-clip: text !important;
    background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    border-radius: 0 !important;
}

.sidebar-section-label { color: #ff2d78 !important; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; padding: 14px 20px 4px 20px; display: block; }
.sidebar-item { display: block; padding: 7px 20px; color: white !important; font-size: 14px; }

.topnav { background-color: #1a1a2e; padding: 12px 40px; display: flex; justify-content: flex-end; align-items: center; gap: 30px; margin-bottom: 20px; border-radius: 30px; }
.topnav a { color: white; text-decoration: none; font-size: 14px; }
.topnav .login-btn { border: 1px solid #555; color: white; padding: 7px 22px; border-radius: 20px; font-size: 14px; background: transparent; cursor: pointer; }
.topnav .signup-btn { background-color: #ff2d78; color: white; padding: 8px 22px; border-radius: 20px; font-weight: 600; font-size: 14px; border: none; cursor: pointer; }

.hero-container { position: relative; width: 100%; height: 340px; overflow: hidden; }
.hero-container img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hero-overlay { position: absolute; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.55); }
.hero-text { position: absolute; bottom: 40px; left: 50px; color: white; font-size: 36px; font-weight: 800; line-height: 1.2; }
.hero-text .pink { color: #ff2d78; }
.input-section { padding: 30px 50px 10px 50px; }
.input-label { color: white; font-size: 22px; font-weight: 700; margin-bottom: 16px; }
.input-label .pink { color: #ff2d78; }
.cards-section { padding: 10px 50px 40px 50px; display: flex; gap: 24px; }
.card { flex: 1; border-radius: 16px; padding: 28px 26px; }
.card-green { background-color: #2d5a3d; }
.card-teal { background-color: #2d4a6b; }
.card h3 { color: white; font-size: 20px; font-weight: 700; margin: 0 0 12px 0; }
.card p { color: white; font-size: 14px; line-height: 1.6; margin: 0; }

div[data-testid="stTextInput"] input { background-color: #1e1e1e !important; color: white !important; border: 1px solid #333 !important; border-radius: 10px !important; font-size: 14px !important; padding: 14px 18px !important; }

.venue-header { text-align: center; margin-bottom: 24px; padding-top: 20px; }
.venue-header h1 { color: white; font-size: 32px; font-weight: 800; margin: 0; }
.venue-header p { color: #aaaaaa; font-size: 14px; margin: 4px 0 0 0; }
.metric-card { background-color: #2a2a2a; border-radius: 14px; padding: 24px; text-align: center; }
.metric-card .metric-value { color: #ff2d78; font-size: 40px; font-weight: 800; margin: 0; }
.metric-card .metric-label { color: #ff2d78; font-size: 14px; font-weight: 600; margin: 4px 0 0 0; }
.box { background-color: #2a2a2a; border-radius: 14px; padding: 20px; margin-bottom: 16px; }
.box-title { color: #ff2d78; font-size: 18px; font-weight: 700; text-align: center; margin-bottom: 16px; }
.vibe-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.vibe-pill { background-color: #3a3a4a; color: white; font-size: 14px; font-weight: 600; border-radius: 10px; padding: 12px; text-align: center; }
.sentiment-row { display: flex; align-items: center; margin-bottom: 10px; gap: 10px; }
.sentiment-label { color: white; font-size: 13px; width: 90px; flex-shrink: 0; }
.sentiment-bar-bg { flex:1; background-color: #444; border-radius: 6px; height: 10px; overflow: hidden; }
.sentiment-bar-fill { height: 100%; border-radius: 6px; }
.sentiment-pct { color: #ff2d78; font-size: 13px; font-weight: 600; width: 38px; text-align: right; flex-shrink: 0; }
.legend { display: flex; gap: 16px; margin-top: 12px; font-size: 12px; color: #aaa; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
.acoustic-row { display: flex; align-items: center; margin-bottom: 10px; gap: 10px; }
.acoustic-label { color: white; font-size: 13px; width: 110px; flex-shrink: 0; }
.acoustic-bar-bg { flex:1; background-color: #444; border-radius: 6px; height: 10px; overflow: hidden; }
.acoustic-bar-fill { height: 100%; border-radius: 6px; background-color: #ff2d78; }
.acoustic-val { color: #dddddd; font-size: 13px; width: 38px; text-align: right; flex-shrink: 0; }

div[data-testid="stSlider"] { padding-top:0 !important; padding-bottom:0 !important; margin-top:-8px !important; margin-bottom:-8px !important; }
div[data-testid="stSlider"] label, div[data-testid="stSlider"] p { color: white !important; font-size: 12px !important; font-family: 'Poppins',sans-serif !important; margin-bottom:0 !important; }
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] { color: #ff2d78 !important; font-weight:700; font-size:11px !important; }
.stSlider div[role="slider"] { background-color: #ff2d78 !important; border-color: #ff2d78 !important; }
[data-testid="stSlider"] > div > div > div > div { background: #ff2d78 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #2a2a2a !important; border: none !important; border-radius: 0 0 14px 14px !important; padding: 4px 14px 14px 14px !important; margin-bottom: 16px !important; }

.playlist-banner { background: linear-gradient(135deg, #1a237e 0%, #1565c0 60%, #0288d1 100%); padding: 24px 32px; display: flex; align-items: center; gap: 24px; }
.album-art { width: 110px; height: 110px; border-radius: 10px; background: linear-gradient(135deg, #00bcd4, #1565c0); display: flex; flex-direction: column; align-items: center; justify-content: center; flex-shrink: 0; }
.album-art-label { color: white; font-size: 11px; font-weight: 800; text-align: center; line-height: 1.2; padding: 8px; }
.banner-info { flex: 1; }
.banner-title { color: white; font-size: 22px; font-weight: 800; margin: 0 0 6px 0; }
.banner-artists { color: #cdd8e8; font-size: 13px; margin: 0 0 12px 0; }
.banner-meta { display: flex; align-items: center; gap: 20px; }
.banner-meta span { color: white; font-size: 12px; }
.play-all { color: #ff2d78; font-size: 13px; font-weight: 700; margin-left: auto; display: flex; align-items: center; gap: 8px; }
.play-btn { width: 30px; height: 30px; border-radius: 50%; border: 2px solid #ff2d78; display: flex; align-items: center; justify-content: center; color: #ff2d78; font-size: 12px; }
.col-headers { display: grid; grid-template-columns: 40px 60px 1fr 120px 200px 100px; padding: 10px 32px; background-color: #0d0d0d; border-bottom: 1px solid #222; color: #888; font-size: 12px; font-weight: 600; text-transform: uppercase; }
.track-row { display: grid; grid-template-columns: 40px 60px 1fr 120px 200px 100px; padding: 10px 32px; align-items: center; border-bottom: 1px solid #161616; }
.track-row:hover { background-color: #1a1a2e; }
.track-num { color: #888; font-size: 13px; text-align: center; }
.track-thumb { width: 42px; height: 42px; border-radius: 6px; background-color: #333; display: flex; align-items: center; justify-content: center; font-size: 18px; }
.track-info { padding-left: 12px; }
.track-title { color: white; font-size: 13px; font-weight: 600; margin: 0; }
.track-artist { color: #888; font-size: 11px; margin: 2px 0 0 0; }
.track-bpm { color: #ccc; font-size: 13px; text-align: center; }
.track-album { color: #888; font-size: 12px; text-align: center; }
.track-actions { display: flex; align-items: center; gap: 10px; justify-content: flex-end; }
.track-actions .heart { color: #888; font-size: 14px; }
.track-actions .duration { color: #888; font-size: 12px; }
.track-actions .dots { color: #888; font-size: 14px; }
.bottom-fade { height: 80px; background: linear-gradient(to bottom, #0d0d0d, #0d2a4a); }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    if st.button("AtmoSound", key="logo_btn"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown('<span class="sidebar-section-label">Menu</span>', unsafe_allow_html=True)
    if st.button("🏠 Home", key="nav_home"):
        st.session_state.page = "home"
        st.rerun()
    if st.button("ℹ️ About Us", key="nav_about"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown('<span class="sidebar-section-label">Library</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">🕐 Recently Generated</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">▶️ Most Played</span>', unsafe_allow_html=True)

    st.markdown('<span class="sidebar-section-label">Playlist and Favorite</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">♡ Your Favorites</span>', unsafe_allow_html=True)

    st.markdown('<span class="sidebar-section-label">General</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">⚙️ Settings</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-item">🚪 Logout</span>', unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════
if st.session_state.page == "home":

    st.markdown("""
    <div class="topnav">
        <a href="#">About Us</a>
        <a href="#">Contact</a>
        <a href="#">How It Works</a>
        <button class="login-btn">Login</button>
        <button class="signup-btn">Sign Up</button>
    </div>""", unsafe_allow_html=True)

    img_path = "Gym pic.webp"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        img_tag = f'<img src="data:image/webp;base64,{img_b64}" />'
    else:
        img_tag = '<div style="width:100%;height:340px;background:#1a1a2e;display:flex;align-items:center;justify-content:center;color:#555;">[ Hero Image ]</div>'

    st.markdown(f"""
    <div class="hero-container">
        {img_tag}
        <div class="hero-overlay"></div>
        <div class="hero-text">The <span class="pink">Perfect Playlist</span><br>with one click</div>
    </div>
    <div class="input-section">
        <div class="input-label">Just paste your venue's <span class="pink">Google Maps link</span> here!</div>
    </div>""", unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        url = st.text_input("url", placeholder="https://www.google.com/maps/search/cornell+tech+cafe", label_visibility="collapsed")
    with col_btn:
        go = st.button("Generate Playlist", key="gen_btn", use_container_width=True)

    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: #ff2d78 !important; color: white !important;
        border: none !important; border-radius: 10px !important;
        font-size: 15px !important; font-weight: 600 !important;
        padding: 14px !important; position: relative !important; height: auto !important;
    }
    </style>""", unsafe_allow_html=True)

    if go and url.strip():
        st.session_state.page = "statistics"
        st.rerun()

    st.markdown("""
    <div class="cards-section">
        <div class="card card-green">
            <h3>Zero Setup</h3>
            <p>Just paste your Google Maps link. The model reads your venue data and does the rest automatically.</p>
        </div>
        <div class="card card-teal">
            <h3>Curated for You!</h3>
            <p>Music adapts to your venue's changing busyness, adjusting BPM and energy levels throughout the day.</p>
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 2 — STATISTICS
# ══════════════════════════════
elif st.session_state.page == "statistics":

    st.markdown("""
    <div class="topnav">
        <a href="#">About Us</a>
        <a href="#">Contact</a>
        <a href="#">How It Works</a>
        <button class="login-btn">Login</button>
        <button class="signup-btn">Sign Up</button>
    </div>
    <div class="venue-header">
        <h1>Cornell Tech Cafe</h1>
        <p>Campus Cafe &nbsp;—&nbsp; Roosevelt Island, NY &nbsp;—&nbsp; 4.4 stars</p>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card"><p class="metric-value">4.4</p><p class="metric-label">Avg Rating</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><p class="metric-value">312</p><p class="metric-label">Reviews</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><p class="metric-value">89%</p><p class="metric-label">Peak Busyness</p></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("""
        <div class="box">
            <div class="box-title">Vibe Tags</div>
            <div class="vibe-grid">
                <div class="vibe-pill">Focused</div>
                <div class="vibe-pill">Bright</div>
                <div class="vibe-pill">Energetic</div>
                <div class="vibe-pill">Modern</div>
            </div>
        </div>
        <div class="box">
            <div class="box-title">Sentiment Breakdown</div>
            <div class="sentiment-row"><span class="sentiment-label">Atmosphere</span><div class="sentiment-bar-bg"><div class="sentiment-bar-fill" style="width:82%;background:#00c97a;"></div></div><span class="sentiment-pct">82%</span></div>
            <div class="sentiment-row"><span class="sentiment-label">Service</span><div class="sentiment-bar-bg"><div class="sentiment-bar-fill" style="width:71%;background:#ff2d78;"></div></div><span class="sentiment-pct">71%</span></div>
            <div class="sentiment-row"><span class="sentiment-label">Noise level</span><div class="sentiment-bar-bg"><div class="sentiment-bar-fill" style="width:58%;background:#888;"></div></div><span class="sentiment-pct">58%</span></div>
            <div class="sentiment-row"><span class="sentiment-label">Crowd</span><div class="sentiment-bar-bg"><div class="sentiment-bar-fill" style="width:18%;background:#ff2d78;"></div></div><span class="sentiment-pct">18%</span></div>
            <div class="legend">
                <span><span class="legend-dot" style="background:#00c97a;"></span>Positive</span>
                <span><span class="legend-dot" style="background:#888;"></span>Neutral</span>
                <span><span class="legend-dot" style="background:#ff2d78;"></span>Negative</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div style="background:#2a2a2a;border-radius:14px 14px 0 0;padding:16px 20px 10px 20px;"><div class="box-title" style="margin-bottom:0;">Busyness by Hour</div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            for label, val in {"7am":20,"9am":75,"11am":60,"12pm":89,"2pm":65,"4pm":42}.items():
                st.slider(label, 0, 100, val, format="%d%%", key=f"busy_{label}")

        acoustic_data = [("Energy",0.62,0.62),("Valence",0.58,0.58),("Danceability",0.48,0.48),("Tempo (BPM)",133,133/200),("Instrumentalness",0.55,0.55)]
        html = '<div class="box"><div class="box-title">Acoustic Targets</div>'
        for lbl, dv, pct in acoustic_data:
            html += f'<div class="acoustic-row"><span class="acoustic-label">{lbl}</span><div class="acoustic-bar-bg"><div class="acoustic-bar-fill" style="width:{int(pct*100)}%;"></div></div><span class="acoustic-val">{dv}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.button("GO TO PLAYLIST", key="go_playlist", use_container_width=True):
        st.session_state.page = "playlist"
        st.rerun()

    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: #00bcd4 !important; color: white !important;
        font-size: 18px !important; font-weight: 700 !important;
        border-radius: 14px !important; padding: 18px !important;
        letter-spacing: 1px !important; border: none !important;
        position: relative !important; height: auto !important;
    }
    </style>""", unsafe_allow_html=True)

# ══════════════════════════════
# PAGE 3 — PLAYLIST
# ══════════════════════════════
elif st.session_state.page == "playlist":

    col_back, col_mid, col_home = st.columns([1, 10, 1])
    with col_back:
        if st.button("← Back", key="back_btn"):
            st.session_state.page = "statistics"
            st.rerun()
    with col_home:
        if st.button("🏠", key="playlist_home"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("""
    <style>
    div[data-testid="stMainBlockContainer"] .stButton > button {
        background-color: transparent !important; color: white !important;
        border: none !important; font-size: 16px !important;
        font-weight: 600 !important; padding: 8px 12px !important;
        position: relative !important; height: auto !important;
        border-radius: 8px !important;
    }
    div[data-testid="stMainBlockContainer"] .stButton > button:hover {
        background-color: #ff2d78 !important;
    }
    </style>
    <div style="background: linear-gradient(135deg, #1a237e 0%, #1565c0 100%); height: 4px;"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="playlist-banner">
        <div class="album-art">
            <div class="album-art-label">TRENDING<br>MUSIC</div>
            <div style="font-size:28px;">🎵</div>
        </div>
        <div class="banner-info">
            <p class="banner-title">The perfect mix for Cornell Tech Cafe</p>
            <p class="banner-artists">tate mcrae, nightmares, the neighbourhood, doja cat and ...</p>
            <div class="banner-meta">
                <span>20 songs</span><span>1h 36m</span><span>1:00pm – 2:36pm</span>
                <div class="play-all">Play All <div class="play-btn">&#9654;</div></div>
            </div>
        </div>
    </div>
    <div class="col-headers">
        <div></div><div></div>
        <div style="padding-left:12px;">Title</div>
        <div style="text-align:center;">BPM</div>
        <div style="text-align:center;">Album</div>
        <div style="text-align:right;">Time</div>
    </div>""", unsafe_allow_html=True)

    tracks = [
        ("Sorfcore","The Neighbourhood",85,"Hard to Imagine Neighbourhood Ever Changing","3:26","🎸"),
        ("Skyfall Beats","nightmares",90,"nightmares","2:45","🎧"),
        ("Greedy","tate mcrae",94,"Greedy","2:11","💜"),
        ("Lovin On me","jack harlow",85,"Lovin On me","2:18","🎤"),
        ("pain the town red","Doja Cat",78,"Paint The Town Red","3:51","🔴"),
        ("Dancin On Night","Buenas",88,"Dance The Night (From Barbie Movie)","2:56","🪩"),
        ("Water","Tyla",92,"Water","3:20","💧"),
        ("Push your limits","Brian michael",86,"Push your limits","2:24","⚡"),
        ("Houdini","Dualipa",95,"Houdini","3:05","🎩"),
        ("Lala","myke towers",82,"La vida es una","3:17","🌟"),
        ("I Wanna Be Yours","arctic monkeys",76,"AM","3:03","🎸"),
        ("Paradise","basement",84,"Paradise","2:38","🌴"),
        ("As It Was","Harry Styles",88,"As It Was","2:47","🌈"),
        ("Another Love","Tom Odell",72,"Another Love","4:06","❤️"),
        ("Daylight","david kushner",80,"Daylight","3:02","☀️"),
        ("Beggin","Maneskin",91,"Chosen","3:21","🎵"),
        ("What Was I Made For","Billie Eilish",83,"What Was I Made For","3:42","🎀"),
        ("Daddy Issues","The Neighbourhood",77,"Wiped out","4:28","🖤"),
        ("Rolling In The Deep","Adele",105,"Adele 21","3:48","🌊"),
        ("OneShot","nhap",88,"Toca Donka","1:15","🎯"),
    ]

    rows_html = ""
    for i, (title, artist, bpm, album, dur, emoji) in enumerate(tracks):
        rows_html += f"""
        <div class="track-row">
            <div class="track-num">{i+1}</div>
            <div class="track-thumb">{emoji}</div>
            <div class="track-info">
                <p class="track-title">{title}</p>
                <p class="track-artist">{artist}</p>
            </div>
            <div class="track-bpm">{bpm}</div>
            <div class="track-album">{album}</div>
            <div class="track-actions">
                <span class="heart">♡</span>
                <span class="duration">{dur}</span>
                <span class="dots">···</span>
            </div>
        </div>"""

    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown('<div class="bottom-fade"></div>', unsafe_allow_html=True)