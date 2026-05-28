"""app.py AI Fitness Trainer  |  Streamlit + YOLOv8 Pose"""
import time, cv2, streamlit as st, numpy as np
@st.cache_resource
def get_model():
    return load_model("yolov8n-pose.pt")
# ── Module-level camera store (survives Streamlit reruns, no serialization) ──
_CAM: dict = {}   # key: camera_id → cv2.VideoCapture

def _open_camera(cam_id: int = 0) -> cv2.VideoCapture:
    """Open camera with DirectShow backend (Windows) and return cap."""
    if cam_id in _CAM and _CAM[cam_id].isOpened():
        return _CAM[cam_id]
    # Try DirectShow first (Windows), fall back to default
    for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(cam_id, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            _CAM[cam_id] = cap
            return cap
        cap.release()
    return None

def _close_camera(cam_id: int = 0):
    if cam_id in _CAM:
        try: _CAM[cam_id].release()
        except: pass
        del _CAM[cam_id]
from yolo_pose import load_model, extract_keypoints, draw_angle_arc
from exercises import analyze_exercise, ExerciseState, TIME_BASED, YOGA_ANALYZERS

st.set_page_config(page_title="AI Fitness Trainer", page_icon="🏋️", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#f0f4ff 0%,#faf5ff 50%,#f0fff4 100%);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.5rem!important;}
.hero{background:linear-gradient(135deg,#667eea,#764ba2);border-radius:24px;padding:36px 48px;
  margin-bottom:32px;box-shadow:0 20px 60px rgba(102,126,234,.35);
  display:flex;align-items:center;justify-content:space-between;}
.hero h1{color:white;font-size:2.4rem;font-weight:800;margin:0;letter-spacing:-1px;}
.hero p{color:rgba(255,255,255,.82);font-size:1.05rem;margin:6px 0 0;}
.sec-title{font-size:1.35rem;font-weight:700;color:#2d3748;margin:0 0 18px;
  padding-bottom:8px;border-bottom:3px solid #667eea;}
.ex-card{background:white;border-radius:18px;box-shadow:0 4px 20px rgba(0,0,0,.08);
  padding:20px 16px;text-align:center;border:2px solid transparent;}
.ex-card .emoji{font-size:2.8rem;margin-bottom:10px;}
.ex-card .name{font-weight:600;font-size:.95rem;color:#2d3748;}
.ex-card .badge{font-size:.7rem;color:#667eea;font-weight:500;background:#eef2ff;
  border-radius:20px;padding:2px 8px;margin-top:6px;display:inline-block;}
.detail-header{background:linear-gradient(135deg,#667eea,#764ba2);color:white;
  border-radius:20px;padding:24px 32px;margin-bottom:24px;
  box-shadow:0 12px 40px rgba(102,126,234,.3);}
.detail-header h2{margin:0 0 4px;font-size:1.8rem;font-weight:800;}
.detail-header p{margin:0;opacity:.85;font-size:.95rem;}
.info-card{background:white;border-radius:16px;padding:20px 22px;margin-bottom:16px;
  box-shadow:0 4px 18px rgba(0,0,0,.07);}
.info-card h4{margin:0 0 10px;font-size:1rem;font-weight:700;color:#4a5568;}
.info-card ul,.info-card ol{margin:0;padding-left:18px;}
.info-card li{margin-bottom:5px;color:#555;font-size:.88rem;}
.stat-box{background:white;border-radius:16px;box-shadow:0 4px 18px rgba(0,0,0,.08);
  padding:20px;text-align:center;}
.stat-label{font-size:.75rem;font-weight:600;color:#718096;text-transform:uppercase;letter-spacing:.5px;}
.stat-value{font-size:2.4rem;font-weight:800;color:#2d3748;line-height:1.1;}
.stat-unit{font-size:.8rem;color:#a0aec0;}
.feedback-ok{background:linear-gradient(90deg,#e8f5e9,#f1f8e9);border-left:4px solid #43a047;
  color:#2e7d32;border-radius:10px;padding:12px 18px;font-weight:600;margin:12px 0;}
.feedback-bad{background:linear-gradient(90deg,#fff3e0,#fff8e1);border-left:4px solid #fb8c00;
  color:#e65100;border-radius:10px;padding:12px 18px;font-weight:600;margin:12px 0;}
.cam-placeholder{background:#f7f8fc;border:2px dashed #cbd5e0;border-radius:18px;
  padding:60px 20px;text-align:center;color:#a0aec0;font-size:1rem;}
div[data-testid="stButton"]>button{border-radius:12px;font-weight:600;font-size:.9rem;
  padding:.55rem 1.4rem;transition:all .2s;}
.yoga-card{background:white;border-radius:16px;box-shadow:0 4px 18px rgba(0,0,0,.08);
  padding:18px;text-align:center;border:2px solid transparent;}
.yoga-card .pose-emoji{font-size:2.4rem;margin-bottom:8px;}
.yoga-card .pose-name{font-weight:700;font-size:.92rem;color:#2d3748;}
.yoga-card .pose-tip{font-size:.75rem;color:#667eea;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

# ── Exercise metadata ─────────────────────────────────────────────────────
EXERCISES = {
    "Pushups":{"emoji":"💪","type":"Reps","desc":"Build upper-body strength targeting chest, shoulders & triceps.","muscles":"Chest · Shoulders · Triceps",
      "instructions":["Start in high plank, hands shoulder-width apart.","Lower chest to nearly touch floor.","Keep core tight & back flat.","Push through palms back to start.","Exhale up, inhale down."],
      "tips":["Elbows ~45° from torso","Don't let hips sag","Full range of motion"],
      "mistakes":["Flared elbows","Partial reps","Head dropping forward"]},
    "Squats":{"emoji":"🏋️","type":"Reps","desc":"Strengthen quads, hamstrings & glutes with proper knee tracking.","muscles":"Quads · Hamstrings · Glutes",
      "instructions":["Feet shoulder-width apart, toes slightly out.","Push hips back & bend knees.","Lower until thighs are parallel to floor.","Keep chest tall, knees over toes.","Drive through heels to stand."],
      "tips":["Knees over toes","Chest up, core braced","Drive through heels"],
      "mistakes":["Knees caving inward","Heels lifting","Rounding lower back"]},
    "Situps":{"emoji":"🔥","type":"Reps","desc":"Classic core exercise for rectus abdominis.","muscles":"Abs · Hip Flexors",
      "instructions":["Lie flat, knees bent, feet flat.","Arms crossed or hands behind head lightly.","Engage core & lift torso toward knees.","Control the descent back to floor.","Don't pull on your neck."],
      "tips":["Chin off chest","Slow controlled descent","Breathe out on the way up"],
      "mistakes":["Pulling on neck","Using momentum","Not going full range"]},
    "Crunches":{"emoji":"⚡","type":"Reps","desc":"Targeted ab contraction for upper abdominal definition.","muscles":"Upper Abs · Core",
      "instructions":["Lie flat, knees bent.","Fingertips lightly behind head.","Curl shoulders off ground.","Hold 1 second at top.","Lower slowly."],
      "tips":["Small controlled movement","Don't pull neck","Feel the ab squeeze"],
      "mistakes":["Full sit-up range","Neck strain","Rushing reps"]},
    "Plank":{"emoji":"🧘","type":"Hold","desc":"Isometric core hold for stability & endurance.","muscles":"Core · Shoulders · Glutes",
      "instructions":["Forearm or high plank position.","Body in straight line head to heels.","Engage core, glutes & legs.","Don't let hips rise or sag.","Breathe steadily."],
      "tips":["Squeeze glutes","Neutral spine","Distribute weight evenly"],
      "mistakes":["Hips too high","Hips sagging","Holding breath"]},
    "Surya Namaskar":{"emoji":"🌅","type":"Cycles","desc":"Sun salutation flow – full-body flexibility & strength.","muscles":"Full Body",
      "instructions":["Stand feet together, hands at chest.","Inhale, raise arms overhead & arch back.","Exhale, fold forward.","Step back to plank, lower to cobra.","Push to downward dog then step forward."],
      "tips":["Match breath to movement","Keep core engaged","Move slowly"],
      "mistakes":["Rushing poses","Holding breath","Collapsing lower back"]},
    "Yoga Poses":{"emoji":"🧘‍♀️","type":"Hold","desc":"Choose from 8 yoga poses for balance, flexibility & mindfulness.","muscles":"Full Body",
      "instructions":["Select a yoga pose from the list.","Set up your foundation carefully.","Engage core gently to stabilize.","Breathe deeply and evenly.","Hold for 30–60 seconds."],
      "tips":["Level shoulders","Breathe steadily","Soft gaze, relaxed jaw"],
      "mistakes":["Holding breath","Over-gripping muscles","Uneven shoulders"]},
}

YOGA_POSES = {
    "Mountain Pose":{"emoji":"🏔️","tip":"Stand tall – align head over hips over ankles",
      "instructions":["Stand with feet together or hip-width.","Arms at sides, palms forward.","Lift through crown of head.","Engage thighs & core gently.","Breathe steadily for 30–60s."],
      "tips":["Feet rooted into floor","Spine tall & long","Shoulders relaxed"]},
    "Warrior I":{"emoji":"⚔️","tip":"Front knee 90°, arms reach overhead",
      "instructions":["Step one foot forward ~4 feet.","Bend front knee to 90°.","Back foot at 45°, heel down.","Raise arms overhead, palms together.","Square hips to the front."],
      "tips":["Front knee over ankle","Hips squared forward","Arms straight up"]},
    "Warrior II":{"emoji":"🛡️","tip":"Arms parallel to floor, front knee bent",
      "instructions":["Wide stance, feet ~4 feet apart.","Front foot forward, back foot 90°.","Bend front knee over ankle.","Arms spread wide parallel to floor.","Gaze over front fingertips."],
      "tips":["Back leg strong & straight","Arms at shoulder height","Sink into front knee"]},
    "Tree Pose":{"emoji":"🌳","tip":"Balance on one leg, other knee out",
      "instructions":["Stand on one foot.","Place other foot on inner thigh/calf.","Hands at chest or arms raised.","Fix gaze on a still point.","Hold 30s then switch."],
      "tips":["Engage standing leg","Don't place foot on knee joint","Core tight"]},
    "Downward Dog":{"emoji":"🐕","tip":"Hips high – inverted V shape",
      "instructions":["Start on hands & knees.","Curl toes under, push hips up & back.","Straighten legs, heels toward floor.","Arms straight, head between arms.","Hold & breathe deeply."],
      "tips":["Push floor away with hands","Hips as high as possible","Heels reaching down"]},
    "Child's Pose":{"emoji":"🧸","tip":"Rest pose – hips to heels, arms forward",
      "instructions":["Kneel, sit back on heels.","Fold torso forward between knees.","Arms stretched forward or at sides.","Forehead on mat.","Breathe deeply into back."],
      "tips":["Hips heavy on heels","Spine long","Totally relax shoulders"]},
    "Cobra Pose":{"emoji":"🐍","tip":"Chest lifted, arms extended, look forward",
      "instructions":["Lie face down, palms under shoulders.","Legs together, tops of feet on floor.","Inhale – lift chest & upper body.","Straighten arms (or keep slight bend).","Shoulders back & down."],
      "tips":["Pubic bone presses floor","Elbows close to body","Don't scrunch neck"]},
    "Triangle Pose":{"emoji":"📐","tip":"Tilt torso sideways, reach toward ankle",
      "instructions":["Wide stance, front foot forward.","Extend arms wide at shoulder height.","Hinge at hip – reach front hand to ankle/shin.","Top arm reaches to ceiling.","Look up toward top hand."],
      "tips":["Both legs straight","Torso opens to side","Don't collapse chest"]},
}

# ── Session state ─────────────────────────────────────────────────────────
for k,v in {"page":"home","selected_exercise":None,"selected_yoga":None,
            "running":False,"ex_state":None,
            "session_reps":0,"session_time":0.0,"session_done":False}.items():
    if k not in st.session_state: st.session_state[k]=v

def go_home():
    _release_cap()
    st.session_state.update(page="home",running=False,session_done=False,selected_yoga=None)

def go_detail(name):
    _release_cap()
    st.session_state.update(page="detail",selected_exercise=name,running=False,
        ex_state=ExerciseState(),session_reps=0,session_time=0.0,session_done=False,selected_yoga=None)

def go_yoga_select():
    _release_cap()
    st.session_state.update(page="yoga_select",running=False,session_done=False)

def go_yoga_pose(pose_name):
    _release_cap()
    st.session_state.update(page="detail",selected_exercise=pose_name,running=False,
        ex_state=ExerciseState(),session_reps=0,session_time=0.0,session_done=False,selected_yoga=pose_name)

def _release_cap():
    _close_camera(0)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 – Home
# ══════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""<div class="hero"><div>
      <h1>🏋️ AI Fitness Trainer</h1>
      <p>Real-time pose detection powered by YOLOv8 · Track reps, form & holds</p>
    </div><div style="font-size:4rem"></div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Choose Your Exercise</div>', unsafe_allow_html=True)
    cols=st.columns(4)
    for i,name in enumerate(EXERCISES):
        ex=EXERCISES[name]
        with cols[i%4]:
            st.markdown(f'<div class="ex-card"><div class="emoji">{ex["emoji"]}</div>'
                f'<div class="name">{name}</div><div class="badge">{ex["type"]}</div></div>',
                unsafe_allow_html=True)
            if st.button(f"Start {name}",key=f"btn_{name}",use_container_width=True):
                if name=="Yoga Poses": go_yoga_select(); st.rerun()
                else: go_detail(name); st.rerun()
    st.markdown("---")
    st.markdown('<div class="sec-title">About the System</div>', unsafe_allow_html=True)
    for col,(val,lbl,unit) in zip(st.columns(4),[("7","Exercises",""),("17","Keypoints","joints"),("YOLOv8","AI Model","Pose"),("Real-time","Detection","")]):
        with col:
            st.markdown(f'<div class="stat-box"><div class="stat-label">{lbl}</div>'
                f'<div class="stat-value">{val}</div><div class="stat-unit">{unit}</div></div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1.5 – Yoga Pose Selection
# ══════════════════════════════════════════════════════════════════════════
def page_yoga_select():
    if st.button("← Back to Exercises",key="back_yoga"): go_home(); st.rerun()
    st.markdown("""<div class="detail-header">
      <h2>🧘‍♀️ Yoga Poses</h2>
      <p>Choose a pose to practice with real-time AI form detection</p>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Select a Yoga Pose</div>', unsafe_allow_html=True)
    cols=st.columns(4)
    for i,(pose,info) in enumerate(YOGA_POSES.items()):
        with cols[i%4]:
            st.markdown(f'<div class="yoga-card"><div class="pose-emoji">{info["emoji"]}</div>'
                f'<div class="pose-name">{pose}</div>'
                f'<div class="pose-tip">{info["tip"]}</div></div>',
                unsafe_allow_html=True)
            if st.button(f"Practice",key=f"yoga_{pose}",use_container_width=True):
                go_yoga_pose(pose); st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 – Exercise Detail + Live Detection
# ══════════════════════════════════════════════════════════════════════════
def page_detail():
    name=st.session_state.selected_exercise
    is_yoga_pose=name in YOGA_POSES
    is_timed=name in TIME_BASED or is_yoga_pose

    if is_yoga_pose:
        ex_info=YOGA_POSES[name]; ex_desc=ex_info; muscles="Balance · Flexibility"
        back_page="yoga_select"
    else:
        ex_info=EXERCISES[name]; ex_desc=ex_info; muscles=ex_info["muscles"]
        back_page="home"

    # Back button
    if st.button(f"← Back",key="back_det"):
        _release_cap(); st.session_state.running=False
        if back_page=="yoga_select": go_yoga_select()
        else: go_home()
        st.rerun()

    emoji=ex_info.get("emoji","🧘")
    desc=ex_info.get("desc",ex_info.get("tip",""))
    st.markdown(f'<div class="detail-header"><h2>{emoji} {name}</h2>'
        f'<p>{desc} &nbsp;|&nbsp; 🎯 {muscles}</p></div>', unsafe_allow_html=True)

    left,right=st.columns([1,1.8],gap="large")

    with left:
        import os
        png_path = f"images/{name.lower().replace(' ', '_')}.png"
        
        if os.path.exists(png_path):
            st.image(png_path, caption=f"Correct Form: {name}", use_column_width=True)
            
        with st.expander(f"Change {name} Image"):
            uploaded_file = st.file_uploader(f"Upload new image for {name}", type=["png", "jpg", "jpeg"], key=f"up_{name}")
            if uploaded_file is not None:
                with open(png_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.rerun()
            
        instructions=ex_info.get("instructions",[])
        steps="".join(f"<li style='margin-bottom:6px'>{s}</li>" for s in instructions)
        st.markdown(f'<div class="info-card"><h4>📋 How to Perform</h4><ol style="padding-left:18px;color:#555;font-size:.88rem">{steps}</ol></div>',unsafe_allow_html=True)
        tips=ex_info.get("tips",[])
        st.markdown('<div class="info-card"><h4>✅ Key Tips</h4><ul>'
            +"".join(f"<li>{t}</li>" for t in tips)+"</ul></div>",unsafe_allow_html=True)
        if not is_yoga_pose:
            mistakes=ex_info.get("mistakes",[])
            st.markdown('<div class="info-card"><h4>⚠️ Common Mistakes</h4>'
                '<ul style="color:#e65100">'+"".join(f"<li>{m}</li>" for m in mistakes)
                +"</ul></div>",unsafe_allow_html=True)

    with right:
        sc1,sc2,sc3=st.columns(3)
        rep_ph=sc1.empty(); time_ph=sc2.empty(); stage_ph=sc3.empty()
        fb_ph=st.empty()
        bc1,bc2=st.columns(2)
        with bc1:
            start_lbl="■ Stop Detection" if st.session_state.running else "▶ Start Detection"
            if st.button(start_lbl,key="toggle",use_container_width=True):
                if st.session_state.running:
                    st.session_state.running=False
                    s=st.session_state.ex_state
                    st.session_state.session_reps=s.reps if s else 0
                    st.session_state.session_time=s.hold_time if s else 0.0
                    st.session_state.session_done=True
                    _release_cap()
                else:
                    st.session_state.running=True
                st.rerun()
        with bc2:
            if st.button("↺ Reset",key="reset",use_container_width=True):
                st.session_state.ex_state=ExerciseState()
                st.session_state.running=False
                st.session_state.session_done=False
                _release_cap(); st.rerun()

        frame_ph=st.empty()

        if not st.session_state.running:
            frame_ph.markdown('<div class="cam-placeholder">📷 Press <b>Start Detection</b> to activate webcam</div>',unsafe_allow_html=True)
            if st.session_state.session_done:
                st.markdown("---")
                st.markdown('<div class="sec-title">📊 Session Summary</div>',unsafe_allow_html=True)
                sa,sb=st.columns(2)
                val=(f"{st.session_state.session_time:.0f}s" if is_timed else str(st.session_state.session_reps))
                lbl="Hold Time" if is_timed else "Total Reps"
                with sa: st.markdown(f'<div class="stat-box"><div class="stat-label">{lbl}</div><div class="stat-value">{val}</div></div>',unsafe_allow_html=True)
                with sb: st.markdown(f'<div class="stat-box"><div class="stat-label">Exercise</div><div class="stat-value" style="font-size:1.4rem">{emoji} {name}</div></div>',unsafe_allow_html=True)
        else:
            # ── continuous webcam loop ───────────────────────
            model = get_model()
            state=st.session_state.ex_state
            if state is None:
                state=ExerciseState(); st.session_state.ex_state=state

            cap=_open_camera(0)
            if cap is None:
                frame_ph.error("❌ Could not open webcam. Check that no other app is using it.")
                st.session_state.running=False
                st.rerun()
            else:
                while st.session_state.running:
                    ret,frame=cap.read()
                    if not ret:
                        frame_ph.warning("⚠️ Camera opened but no frame received. Retrying…")
                        time.sleep(0.1)
                        continue
                        
                    frame=cv2.flip(frame,1)
                    frame = cv2.resize(frame, (640, 480))
                    kps,annotated=extract_keypoints(frame,model)
                    result={"reps":state.reps,"stage":state.stage,
                            "feedback":"🔍 Get into position…","correct":False,
                            "angle":None,"angle_pt":None,"hold_time":state.hold_time}
                    if kps is not None:
                        result=analyze_exercise(name,kps,state)
                        if result["angle"] is not None and result["angle_pt"] is not None:
                            draw_angle_arc(annotated,result["angle_pt"],result["angle"])
                    _draw_hud(annotated,result,name,is_timed)
                    rgb=cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)
                    frame_ph.image(rgb, channels="RGB")

                    rep_val=f"{result['reps']}" if not is_timed else "—"
                    time_val=f"{result.get('hold_time',0):.1f}s" if is_timed else "—"
                    stage_val=str(result.get("stage") or "—").upper()
                    rep_ph.markdown(f'<div class="stat-box"><div class="stat-label">{"Reps" if not is_timed else "Cycles"}</div><div class="stat-value">{rep_val}</div></div>',unsafe_allow_html=True)
                    time_ph.markdown(f'<div class="stat-box"><div class="stat-label">Hold Time</div><div class="stat-value">{time_val}</div></div>',unsafe_allow_html=True)
                    stage_ph.markdown(f'<div class="stat-box"><div class="stat-label">Stage</div><div class="stat-value" style="font-size:1rem">{stage_val}</div></div>',unsafe_allow_html=True)
                    cls="feedback-ok" if result["correct"] else "feedback-bad"
                    fb_ph.markdown(f'<div class="{cls}">{result["feedback"]}</div>',unsafe_allow_html=True)

                    time.sleep(0.03)

def _draw_hud(frame,result,name,is_timed):
    h,w=frame.shape[:2]
    overlay=frame.copy()
    cv2.rectangle(overlay,(0,0),(w,70),(30,30,50),-1)
    cv2.addWeighted(overlay,.65,frame,.35,0,frame)
    cv2.putText(frame,name,(14,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA)
    val=f"Hold: {result.get('hold_time',0):.1f}s" if is_timed else f"Reps: {result['reps']}"
    cv2.putText(frame,val,(14,58),cv2.FONT_HERSHEY_SIMPLEX,.65,(180,230,255),2,cv2.LINE_AA)
    color=(50,200,80) if result["correct"] else (30,150,255)
    label="Correct Form ✓" if result["correct"] else "Adjust Posture"
    cv2.circle(frame,(w-20,20),10,color,-1,cv2.LINE_AA)
    cv2.putText(frame,label,(w-155,25),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)

# ── Router ────────────────────────────────────────────────────────────────
page=st.session_state.page
if page=="home": page_home()
elif page=="yoga_select": page_yoga_select()
else: page_detail()
