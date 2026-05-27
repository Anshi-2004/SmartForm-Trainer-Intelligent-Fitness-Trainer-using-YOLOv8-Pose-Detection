"""exercises.py – Rep counting, form checks, yoga pose detection."""
import time, numpy as np
from pose_utils import get_keypoint, calculate_angle, all_visible, midpoint

# ── helpers ───────────────────────────────────────────────────────────────
def _ang(kps, a, b, c):
    A,B,C = get_keypoint(kps,a), get_keypoint(kps,b), get_keypoint(kps,c)
    if not all_visible(A,B,C,threshold=0.25): return None,None
    return calculate_angle(A,B,C), B

class ExerciseState:
    def __init__(self):
        self.reps=0; self.stage=None; self.timer_start=None; self.hold_time=0.0
    def reset(self): self.__init__()

def _no(name):
    return {"reps":0,"stage":None,"feedback":"🔍 No pose detected – stand in frame","correct":False,"angle":None,"angle_pt":None,"hold_time":0.0}

# ── rep-based ─────────────────────────────────────────────────────────────
def analyze_pushup(kps,s):
    la,lp=_ang(kps,"left_shoulder","left_elbow","left_wrist")
    ra,rp=_ang(kps,"right_shoulder","right_elbow","right_wrist")
    if la is None and ra is None: return _no("Pushup")
    angle=float(np.mean([a for a in [la,ra] if a is not None])); pt=lp or rp
    if angle<90: s.stage="down"; fb="Going down – keep back flat! 💪"; ok=True
    elif angle>160:
        if s.stage=="down": s.reps+=1
        s.stage="up"; fb="Arms extended – great rep! ✅"; ok=True
    else: fb="Keep going…"; ok=True
    ls,lh,lk=get_keypoint(kps,"left_shoulder"),get_keypoint(kps,"left_hip"),get_keypoint(kps,"left_knee")
    if ls and lh and lk and all_visible(ls,lh,lk) and calculate_angle(ls,lh,lk)<155:
        fb="⚠️ Keep back straight!"; ok=False
    return {"reps":s.reps,"stage":s.stage,"feedback":fb,"correct":ok,"angle":angle,"angle_pt":pt,"hold_time":0.0}

def analyze_squat(kps,s):
    la,lp=_ang(kps,"left_hip","left_knee","left_ankle")
    ra,rp=_ang(kps,"right_hip","right_knee","right_ankle")
    if la is None and ra is None: return _no("Squat")
    angle=float(np.mean([a for a in [la,ra] if a is not None])); pt=lp or rp
    if angle<90: s.stage="down"; fb="In squat – hold! 🏋️"; ok=True
    elif angle>160:
        if s.stage=="down": s.reps+=1
        s.stage="up"; fb="Rep complete! ✅"; ok=True
    else: fb="Descending – knees over toes 👍"; ok=True
    return {"reps":s.reps,"stage":s.stage,"feedback":fb,"correct":ok,"angle":angle,"angle_pt":pt,"hold_time":0.0}

def analyze_situp(kps,s):
    la,lp=_ang(kps,"left_shoulder","left_hip","left_knee")
    ra,rp=_ang(kps,"right_shoulder","right_hip","right_knee")
    if la is None and ra is None: return _no("Situp")
    angle=float(np.mean([a for a in [la,ra] if a is not None])); pt=lp or rp
    if angle<55: s.stage="up"; fb="Sit up – nice! ✅"; ok=True
    elif angle>120:
        if s.stage=="up": s.reps+=1
        s.stage="down"; fb="Lowering – controlled 💪"; ok=True
    else: fb="Keep going!"; ok=True
    return {"reps":s.reps,"stage":s.stage,"feedback":fb,"correct":ok,"angle":angle,"angle_pt":pt,"hold_time":0.0}

def analyze_crunch(kps,s):
    la,lp=_ang(kps,"left_shoulder","left_hip","left_knee")
    ra,rp=_ang(kps,"right_shoulder","right_hip","right_knee")
    if la is None and ra is None: return _no("Crunch")
    angle=float(np.mean([a for a in [la,ra] if a is not None])); pt=lp or rp
    if angle<80: s.stage="up"; fb="Crunch – feel the burn! 🔥"; ok=True
    elif angle>130:
        if s.stage=="up": s.reps+=1
        s.stage="down"; fb="Release slowly"; ok=True
    else: fb="Push through!"; ok=True
    return {"reps":s.reps,"stage":s.stage,"feedback":fb,"correct":ok,"angle":angle,"angle_pt":pt,"hold_time":0.0}

def analyze_surya(kps,s):
    la,lp=_ang(kps,"left_shoulder","left_elbow","left_wrist")
    ra,rp=_ang(kps,"right_shoulder","right_elbow","right_wrist")
    if la is None and ra is None: return _no("Surya")
    angle=float(np.mean([a for a in [la,ra] if a is not None])); pt=lp or rp
    if angle>155: s.stage="raised"; fb="Arms raised – Namaste! 🙏"; ok=True
    elif angle<80:
        if s.stage=="raised": s.reps+=1
        s.stage="lowered"; fb="Flow into next pose"; ok=True
    else: fb="Flowing…"; ok=True
    return {"reps":s.reps,"stage":s.stage,"feedback":fb,"correct":ok,"angle":angle,"angle_pt":pt,"hold_time":0.0}

# ── time-based ────────────────────────────────────────────────────────────
def analyze_plank(kps,s):
    ls,lh,la=get_keypoint(kps,"left_shoulder"),get_keypoint(kps,"left_hip"),get_keypoint(kps,"left_ankle")
    if not all_visible(ls,lh,la,threshold=0.25): return _no("Plank")
    ang=calculate_angle(ls,lh,la); ok=155<ang<205
    now=time.time()
    if ok:
        if s.timer_start is None: s.timer_start=now
        s.hold_time=now-s.timer_start; fb=f"Plank held! {int(ang)}° ✅"
    else:
        s.timer_start=None
        fb="⚠️ Raise hips – body must be straight!" if ang<155 else "⚠️ Lower hips!"
    return {"reps":0,"stage":"hold" if ok else "broken","feedback":fb,"correct":ok,"angle":ang,"angle_pt":lh,"hold_time":s.hold_time}

# ── individual yoga poses ─────────────────────────────────────────────────
def _hold_result(s,ok,fb,ang,pt):
    now=time.time()
    if ok:
        if s.timer_start is None: s.timer_start=now
        s.hold_time=now-s.timer_start
    else:
        s.timer_start=None
    return {"reps":0,"stage":"hold" if ok else "broken","feedback":fb,"correct":ok,"angle":ang,"angle_pt":pt,"hold_time":s.hold_time}

def yoga_mountain(kps,s):
    """Tadasana – upright, arms at sides, body aligned."""
    ls,lh,la=get_keypoint(kps,"left_shoulder"),get_keypoint(kps,"left_hip"),get_keypoint(kps,"left_ankle")
    if not all_visible(ls,lh,la,threshold=0.25): return _no("Mountain")
    ang=calculate_angle(ls,lh,la); ok=165<ang<195
    fb=("Mountain pose – tall and steady! 🏔️ ✅" if ok else "⚠️ Stand upright – align head over hips")
    return _hold_result(s,ok,fb,ang,lh)

def yoga_warrior1(kps,s):
    """Warrior I – front knee ~90°, arms raised overhead."""
    ka,kp=_ang(kps,"left_hip","left_knee","left_ankle")
    aa,ap=_ang(kps,"left_shoulder","left_elbow","left_wrist")
    if ka is None: return _no("Warrior I")
    knee_ok=80<ka<110; arms_ok=aa is not None and aa>140
    ok=knee_ok and arms_ok
    if not knee_ok: fb=f"⚠️ Bend front knee to ~90° (now {int(ka)}°)"
    elif not arms_ok: fb="⚠️ Raise arms fully overhead"
    else: fb="Warrior I – strong stance! ⚔️ ✅"
    return _hold_result(s,ok,fb,ka,kp)

def yoga_warrior2(kps,s):
    """Warrior II – front knee bent, arms horizontal."""
    ka,kp=_ang(kps,"left_hip","left_knee","left_ankle")
    # Check arm horizontality via shoulder-wrist y difference
    ls,lw=get_keypoint(kps,"left_shoulder"),get_keypoint(kps,"left_wrist")
    rs,rw=get_keypoint(kps,"right_shoulder"),get_keypoint(kps,"right_wrist")
    if ka is None: return _no("Warrior II")
    knee_ok=80<ka<110
    arm_ok=(ls and lw and abs(ls[1]-lw[1])<60) or (rs and rw and abs(rs[1]-rw[1])<60)
    ok=knee_ok and arm_ok
    if not knee_ok: fb=f"⚠️ Bend front knee (~90°, now {int(ka)}°)"
    elif not arm_ok: fb="⚠️ Extend arms parallel to floor"
    else: fb="Warrior II – arms spread wide! ✅"
    return _hold_result(s,ok,fb,ka,kp)

def yoga_tree(kps,s):
    """Vrikshasana – balance on one leg, other knee out."""
    lk,lh,la=get_keypoint(kps,"left_knee"),get_keypoint(kps,"left_hip"),get_keypoint(kps,"left_ankle")
    rk=get_keypoint(kps,"right_knee")
    if not all_visible(lk,lh,la,threshold=0.25): return _no("Tree")
    # Standing leg straight, raised leg knee bent outward
    stand_ang,sp=_ang(kps,"left_hip","left_knee","left_ankle")
    if stand_ang is None: return _no("Tree")
    ok=stand_ang>155 and rk is not None and rk[0]<lk[0]-30
    fb=("Tree pose – balanced! 🌳 ✅" if ok else "⚠️ Straighten standing leg & lift other knee out")
    return _hold_result(s,ok,fb,stand_ang,sp)

def yoga_downdog(kps,s):
    """Downward Dog – hips high, body inverted V."""
    ls,lh,la=get_keypoint(kps,"left_shoulder"),get_keypoint(kps,"left_hip"),get_keypoint(kps,"left_ankle")
    if not all_visible(ls,lh,la,threshold=0.25): return _no("Downdog")
    ang=calculate_angle(ls,lh,la)
    # Hip should be the highest point – hip y < shoulder y and hip y < ankle y (y increases downward)
    hip_high=(lh[1]<ls[1]) and (lh[1]<la[1])
    ok=80<ang<130 and hip_high
    fb=("Downward Dog – hips high! 🐕 ✅" if ok else "⚠️ Push hips up & back – form inverted V")
    return _hold_result(s,ok,fb,ang,lh)

def yoga_child(kps,s):
    """Child's Pose – deep forward fold, hips back."""
    ls,lh,lk=get_keypoint(kps,"left_shoulder"),get_keypoint(kps,"left_hip"),get_keypoint(kps,"left_knee")
    if not all_visible(ls,lh,lk,threshold=0.25): return _no("Child")
    ang=calculate_angle(ls,lh,lk); ok=ang<70
    fb=("Child's Pose – rest deeply! 🧸 ✅" if ok else "⚠️ Fold forward more – bring chest to floor")
    return _hold_result(s,ok,fb,ang,lh)

def yoga_cobra(kps,s):
    """Cobra Pose – upper body lifted, elbows extended."""
    ls,le,lw=get_keypoint(kps,"left_shoulder"),get_keypoint(kps,"left_elbow"),get_keypoint(kps,"left_wrist")
    if not all_visible(ls,le,lw,threshold=0.25): return _no("Cobra")
    ang,pt=_ang(kps,"left_shoulder","left_elbow","left_wrist")
    if ang is None: return _no("Cobra")
    # Shoulders should be above hips (y decreases upward on screen)
    lh=get_keypoint(kps,"left_hip")
    chest_up=lh is not None and ls[1]<lh[1]
    ok=ang>140 and chest_up
    fb=("Cobra pose – chest lifted! 🐍 ✅" if ok else "⚠️ Lift chest up, extend arms, look forward")
    return _hold_result(s,ok,fb,ang,pt)

def yoga_triangle(kps,s):
    """Triangle Pose – lateral bend, one arm up one down."""
    ls,lh,la=get_keypoint(kps,"left_shoulder"),get_keypoint(kps,"left_hip"),get_keypoint(kps,"left_ankle")
    if not all_visible(ls,lh,la,threshold=0.25): return _no("Triangle")
    # Check lateral tilt – shoulder and hip not vertically aligned
    ang=calculate_angle(ls,lh,la)
    tilt=abs(ls[0]-lh[0])  # horizontal offset
    ok=tilt>60 and 130<ang<180
    fb=("Triangle pose – great stretch! 📐 ✅" if ok else "⚠️ Tilt torso sideways, reach hand toward ankle")
    return _hold_result(s,ok,fb,ang,lh)

# ── dispatchers ───────────────────────────────────────────────────────────
EXERCISE_ANALYZERS = {
    "Pushups":analyze_pushup,"Squats":analyze_squat,
    "Situps":analyze_situp,"Crunches":analyze_crunch,
    "Plank":analyze_plank,"Surya Namaskar":analyze_surya,
}

YOGA_ANALYZERS = {
    "Mountain Pose":yoga_mountain,"Warrior I":yoga_warrior1,
    "Warrior II":yoga_warrior2,"Tree Pose":yoga_tree,
    "Downward Dog":yoga_downdog,"Child's Pose":yoga_child,
    "Cobra Pose":yoga_cobra,"Triangle Pose":yoga_triangle,
}

TIME_BASED = {"Plank","Yoga Poses"}

def analyze_exercise(name,kps,state):
    if name in YOGA_ANALYZERS: fn=YOGA_ANALYZERS[name]
    else: fn=EXERCISE_ANALYZERS.get(name)
    if fn is None: return _no(name)
    r=fn(kps,state)
    if "hold_time" not in r: r["hold_time"]=0.0
    return r
