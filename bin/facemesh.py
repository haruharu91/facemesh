import sys
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial import distance
import argparse
import hashlib
import base64

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Initialize Original InsightFace Engine (Buffalo_L)
app = FaceAnalysis(
    name="buffalo_l", 
    root=resource_path("."),
    providers=['CoreMLExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))

# --- HUGGING FACE DEMOGRAPHICS IMPORTS (OFFLINE CONFIGURATION) ---
from transformers import pipeline, CLIPProcessor, CLIPModel
from PIL import Image

# Enforce strict offline mode globally
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_LOCAL_PATH = resource_path(os.path.join("models", "clip-vit-base-patch32"))

# Load pipeline strictly from local directory using explicit components
try:
    if os.path.exists(MODEL_LOCAL_PATH):
        processor = CLIPProcessor.from_pretrained(MODEL_LOCAL_PATH, local_files_only=True)
        model = CLIPModel.from_pretrained(MODEL_LOCAL_PATH, local_files_only=True)
        
        hf_classifier = pipeline(
            "zero-shot-image-classification", 
            model=model,
            feature_extractor=processor,
            tokenizer=processor.tokenizer
        )
    else:
        raise FileNotFoundError(f"Local model folder missing at '{MODEL_LOCAL_PATH}'.")
except Exception as e:
    print(f"[Warning]: Failed to load offline HF pipeline: {e}")
    hf_classifier = None

FAIRFACE_LABELS = [
    "White person", "Black person", "Indian person", 
    "East Asian person", "Southeast Asian person", 
    "Middle Eastern person", "Latino or Hispanic person"
]

AGE_LABELS = [
    "a photo of a child", "a photo of a young youth", 
    "a photo of a middle-aged adult", "a photo of an elderly senior citizen"
]

MOOD_LABELS = [
    "a photo of a happy smiling face", "a photo of a neutral calm face",
    "a photo of a serious focused face", "a photo of a sad somber face"
]

def analyze_demographics_huggingface(image_path, include_race=False):
    """ Extracts face crop via InsightFace and runs CLIP classification for Demographics """
    if hf_classifier is None:
        return "Offline", "Offline", "Offline"

    try:
        img = cv2.imread(image_path)
        if img is None: return "Offline", "Offline", "Offline"
        faces = app.get(img)
        if not faces: return "Offline", "Offline", "Offline"
        
        bbox = faces[0].bbox.astype(int)
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
        
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0: return "Offline", "Offline", "Offline"
        
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_crop_rgb)
        
        if include_race:
            race_results = hf_classifier(pil_img, candidate_labels=FAIRFACE_LABELS)
            race_label = race_results[0]['label'].replace(" person", "").title()
            race_conf = race_results[0]['score'] * 100
            race_str = f"{race_label} ({race_conf:.1f}%)"
        else:
            race_str = "Disabled (Compliance Mode)"
        
        age_results = hf_classifier(pil_img, candidate_labels=AGE_LABELS)
        top_age = age_results[0]['label']
        age_map = {
            "a photo of a child": "Child (0-12)",
            "a photo of a young youth": "Youth (13-29)",
            "a photo of a middle-aged adult": "Adult (30-59)",
            "a photo of an elderly senior citizen": "Senior (60+)"
        }
        age_str = age_map.get(top_age, "Adult (30-59)")

        mood_results = hf_classifier(pil_img, candidate_labels=MOOD_LABELS)
        top_mood = mood_results[0]['label']
        mood_map = {
            "a photo of a happy smiling face": "Happy",
            "a photo of a neutral calm face": "Neutral",
            "a photo of a serious focused face": "Serious",
            "a photo of a sad somber face": "Sad"
        }
        mood_str = mood_map.get(top_mood, "Neutral")
        
        return race_str, age_str, mood_str
    except Exception:
        return "Offline", "Offline", "Offline"

def get_pupil_centers(landmarks):
    """ Computes pupil centers based on standard 106-point landmark indices. """
    if len(landmarks) >= 96:
        left_pupil = np.mean(landmarks[35:41], axis=0)
        right_pupil = np.mean(landmarks[89:95], axis=0)
    else:
        left_pupil = np.mean(landmarks[0:4], axis=0)
        right_pupil = np.mean(landmarks[4:8], axis=0)
    return left_pupil, right_pupil

def extract_pitch_normalized_ratios(landmarks, pitch_deg=0.0):
    """ Extracts scale-invariant ratios with 3D pitch correction applied to vertical axes. """
    en_left, en_right = landmarks[35], landmarks[89]      # Inner canthi
    ex_left, ex_right = landmarks[39], landmarks[93]      # Outer canthi
    left_pupil, right_pupil = get_pupil_centers(landmarks) # Pupils
    nasion = landmarks[16]                                # Bony nasal bridge root
    subnasale = landmarks[86]                             # Anterior nasal spine attachment

    w_ex = np.linalg.norm(ex_left - ex_right)
    w_en = np.linalg.norm(en_left - en_right)
    if w_ex == 0 or w_en == 0:
        return {"canthal": 0.0, "pupillary": 0.0, "composite_nasion": 0.0, "subnasale": 0.0}

    pitch_rad = np.radians(abs(pitch_deg))
    cos_correction = max(0.20, np.cos(pitch_rad))

    raw_2d_nasion = np.linalg.norm(nasion - subnasale)
    pitch_corrected_nasion = raw_2d_nasion / cos_correction

    nasion_ratio_ex = pitch_corrected_nasion / w_ex
    nasion_ratio_en = pitch_corrected_nasion / w_en
    composite_nasion_ratio = (nasion_ratio_ex + nasion_ratio_en) / 2.0

    return {
        "canthal": w_en / w_ex,
        "pupillary": np.linalg.norm(left_pupil - right_pupil) / w_ex,
        "composite_nasion": composite_nasion_ratio,
        "subnasale": (np.linalg.norm(landmarks[80] - landmarks[82]) / cos_correction) / w_ex
    }

def compute_skeletal_index_array(landmarks_t, pitch_t, landmarks_s, pitch_s):
    """ Computes individual landmark variances with 3D pitch pose compensation. """
    r_t = extract_pitch_normalized_ratios(landmarks_t, pitch_t)
    r_s = extract_pitch_normalized_ratios(landmarks_s, pitch_s)

    weights_config = [
        ("Intercanthal Ratio (en-en / ex-ex)", "canthal", 0.55, "Very High (Rigid Anchor)"),
        ("Interpupillary Ratio (IPD / ex-ex)", "pupillary", 0.25, "Very High (Orbital Fixed)"),
        ("Composite Nasion Ridge (3D Pitch-Corrected)", "composite_nasion", 0.15, "High (Pitch Normalized)"),
        ("Subnasale Anchor Ratio (Alar Base / ex-ex)", "subnasale", 0.05, "Moderate")
    ]

    skeletal_array = []
    composite_variance_pct = 0.0

    for label, key, weight, inv_level in weights_config:
        var_pct = abs(r_t[key] - r_s[key]) * 100.0
        composite_variance_pct += weight * var_pct
        skeletal_array.append({
            "label": label,
            "weight": weight,
            "variance_pct": var_pct,
            "invariance": inv_level
        })

    skeletal_array.sort(key=lambda x: x["weight"], reverse=True)
    canthal_var_pct = abs(r_t["canthal"] - r_s["canthal"]) * 100.0

    return canthal_var_pct, composite_variance_pct, skeletal_array

def process_face_data(image_path):
    """ Detects faces, extracts embedding, pitch angle, and landmark points """
    abs_image_path = os.path.abspath(image_path)
    img = cv2.imread(abs_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open/read file: {abs_image_path}")
        
    faces = app.get(img)
    if not faces or len(faces) == 0:
        raise ValueError(f"Face Analysis Failed: No face detected in '{os.path.basename(abs_image_path)}'.")
        
    face = faces[0]
    landmarks = None
    if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
        landmarks = face.landmark_2d_106.astype(float)
    elif hasattr(face, 'landmark_2d') and face.landmark_2d is not None:
        landmarks = face.landmark_2d.astype(float)
        
    if landmarks is None:
        raise ValueError("Model attribute mismatch. Landmark keys missing.")

    pitch_angle = float(face.pose[0]) if hasattr(face, 'pose') and face.pose is not None else 0.0
        
    return img, landmarks, face.normed_embedding, pitch_angle

def generate_pupil_aligned_mesh(img, landmarks, target_w, target_h, color, label_text):
    """
    Normalizes, scales, and levels a face mesh canvas.
    Performs 2D similarity transform (translation, scaling, 2D roll angle leveling)
    so both pupils align horizontally on the visual canvas.
    """
    left_pupil, right_pupil = get_pupil_centers(landmarks)
    
    pupil_center = (left_pupil + right_pupil) / 2.0
    pupil_distance = np.linalg.norm(left_pupil - right_pupil)
    
    # Calculate 2D roll angle to level pupils horizontally
    d_y = right_pupil[1] - left_pupil[1]
    d_x = right_pupil[0] - left_pupil[0]
    angle_rad = np.arctan2(d_y, d_x)

    desired_dist = target_w * 0.25
    scale = desired_dist / pupil_distance
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas_center = np.array([target_w / 2.0, target_h / 2.0])
    
    # 1. Translate to origin
    centered = landmarks - pupil_center
    
    # 2. Rotate to level eyes horizontally (2D alignment)
    cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = np.dot(centered, rot_matrix)
    
    # 3. Scale and translate to canvas center
    aligned_landmarks = (rotated * scale) + canvas_center
    
    rect = (0, 0, target_w, target_h)
    subdiv = cv2.Subdiv2D(rect)
    
    for p in aligned_landmarks:
        if 0 <= p[0] < target_w and 0 <= p[1] < target_h:
            subdiv.insert((float(p[0]), float(p[1])))
            
    try:
        triangle_list = subdiv.getTriangleList()
        for t in triangle_list:
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))
            
            if (0 <= pt1[0] < target_w and 0 <= pt1[1] < target_h and
                0 <= pt2[0] < target_w and 0 <= pt2[1] < target_h and
                0 <= pt3[0] < target_w and 0 <= pt3[1] < target_h):
                
                cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
                cv2.line(canvas, pt2, pt3, color, 1, cv2.LINE_AA)
                cv2.line(canvas, pt3, pt1, color, 1, cv2.LINE_AA)
    except Exception:
        pass

    for (x, y) in aligned_landmarks.astype(int):
        if 0 <= x < target_w and 0 <= y < target_h:
            cv2.circle(canvas, (x, y), 2, (0, 255, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, target_h / 850.0)
    thickness = max(1, int(target_h / 450.0))
    
    # Single Header Label
    text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
    x_pos = int((target_w - text_size[0]) / 2)
    y_pos = int(target_h * 0.06)
    cv2.putText(canvas, label_text, (x_pos, y_pos), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, label_text, (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)

    return canvas

def calc_font_scale(text, font, base_scale, thickness, max_width):
    """ Dynamically scales font down to fit strictly inside maximum image boundaries """
    text_w = cv2.getTextSize(text, font, base_scale, thickness)[0][0]
    if text_w > max_width:
        return base_scale * (max_width / float(text_w))
    return base_scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightFace 3D Pitch-Compensated Identity Framework")
    parser.add_argument("target", help="Filename or path of target image")
    parser.add_argument("suspect", help="Filename or path of suspect image")
    parser.add_argument("--detect-race", action="store_true", help="Explicitly enable race tracking metrics")
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "img"))

    target_path = os.path.abspath(args.target) if os.path.exists(args.target) else os.path.join(IMG_DIR, args.target)
    suspect_path = os.path.abspath(args.suspect) if os.path.exists(args.suspect) else os.path.join(IMG_DIR, args.suspect)

    if not os.path.exists(target_path) or not os.path.exists(suspect_path):
        print(f"[Error]: Source files missing.")
        sys.exit(1)

    try:
        print("\n--- Running Dual-Axis Biometric & Skeletal Pipeline ---")
        
        t_img, t_landmarks, vector_target, pitch_t = process_face_data(target_path)
        s_img, s_landmarks, vector_suspect, pitch_s = process_face_data(suspect_path)

        # 1. Face Mesh Cosine Distance (Mask / Persona Match)
        face_mesh_cosine_dist = distance.cosine(vector_target, vector_suspect)

        # 2 & 3. Pitch-Normalised Canthal Variance (%) & Descending Weighted Skeletal Array
        canthal_var_pct, composite_skel_var, skeletal_array = compute_skeletal_index_array(t_landmarks, pitch_t, s_landmarks, pitch_s)

        # Match Likelihood Evaluation
        mask_match_pass = face_mesh_cosine_dist < 0.35
        canthus_match_pass = canthal_var_pct <= 3.0
        skeletal_index_pass = composite_skel_var <= 2.5

        mask_likelihood = "HIGH LIKELIHOOD" if face_mesh_cosine_dist < 0.28 else ("MODERATE LIKELIHOOD" if mask_match_pass else "UNLIKELY")
        canthus_likelihood = "HIGH LIKELIHOOD" if canthal_var_pct <= 1.0 else ("MODERATE LIKELIHOOD" if canthus_match_pass else "UNLIKELY")
        skeletal_likelihood = "HIGH LIKELIHOOD" if composite_skel_var <= 1.2 else ("MODERATE LIKELIHOOD" if skeletal_index_pass else "UNLIKELY")

        # 4. Persona State Analytic Categorization
        if mask_match_pass and skeletal_index_pass:
            persona_state = "STATE 1: AUTHENTIC IDENTITY MATCH (Matching Persona & Skull Structure)"
        elif not mask_match_pass and skeletal_index_pass:
            persona_state = "STATE 2: ACTOR PERSONA / CHARACTER MASK (Distinct Persona Mask, IDENTICAL Skull Body)"
        elif mask_match_pass and not skeletal_index_pass:
            persona_state = "STATE 3: HIGH-FIDELITY DOPPELGANGER / MPO SPOOF (Matching Style, DIFFERENT Skull Body)"
        else:
            persona_state = "STATE 4: DISTINCT INDIVIDUALS (Totally Different People Across All Metrics)"

        # --- RESIZING & RESOLUTION UPSCALING ---
        min_default_height = 2160

        h1, w1 = t_img.shape[:2]
        h2, w2 = s_img.shape[:2]

        # Use the largest dimension or the default floor height
        target_height = max(min_default_height, h1, h2)

        t_scale, s_scale = target_height / h1, target_height / h2
        t_img_res = cv2.resize(t_img, (int(w1 * t_scale), target_height), interpolation=cv2.INTER_CUBIC)
        s_img_res = cv2.resize(s_img, (int(w2 * s_scale), target_height), interpolation=cv2.INTER_CUBIC)

        color_t = (255, 140, 0)   
        color_s = (30, 144, 255)  
        
        # Mesh Generation
        t_mesh_canvas = generate_pupil_aligned_mesh(
            t_img, t_landmarks, t_img_res.shape[1], target_height, color_t, "TARGET FACE MESH"
        )
        s_mesh_canvas = generate_pupil_aligned_mesh(
            s_img, s_landmarks, s_img_res.shape[1], target_height, color_s, "SUSPECT FACE MESH"
        )

        composite_ribbon = np.hstack((t_img_res, t_mesh_canvas, s_img_res, s_mesh_canvas))
        ribbon_w = composite_ribbon.shape[1]

        # --- DUAL HASH GENERATION ---
        success, ribbon_png_bytes = cv2.imencode('.png', composite_ribbon)
        if not success:
            raise RuntimeError("Failed to encode visual ribbon image for hashing.")
        
        sha_img_hex = hashlib.sha256(ribbon_png_bytes.tobytes()).hexdigest()
        sha_img_b64 = base64.urlsafe_b64encode(hashlib.sha256(ribbon_png_bytes.tobytes()).digest()).decode('utf-8').rstrip('=')[:12]

        target_race, target_age, target_mood = analyze_demographics_huggingface(target_path, include_race=args.detect_race)
        suspect_race, suspect_age, suspect_mood = analyze_demographics_huggingface(suspect_path, include_race=args.detect_race)

        target_metrics_str = f"Age: {target_age} | Mood: {target_mood}"
        suspect_metrics_str = f"Age: {suspect_age} | Mood: {suspect_mood}"
        if args.detect_race:
            target_metrics_str = f"Race: {target_race} | " + target_metrics_str
            suspect_metrics_str = f"Race: {suspect_race} | " + suspect_metrics_str

        raw_data_payload = (
            f"TARGET:{os.path.basename(target_path)}|PITCH:{pitch_t:.4f}|"
            f"SUSPECT:{os.path.basename(suspect_path)}|PITCH:{pitch_s:.4f}|"
            f"COS_DIST:{face_mesh_cosine_dist:.6f}|CANTHAL_VAR:{canthal_var_pct:.6f}|"
            f"SKEL_VAR:{composite_skel_var:.6f}"
        )
        sha_data_hex = hashlib.sha256(raw_data_payload.encode('utf-8')).hexdigest()
        sha_data_b64 = base64.urlsafe_b64encode(hashlib.sha256(raw_data_payload.encode('utf-8')).digest()).decode('utf-8').rstrip('=')[:12]

        session_bytes = os.urandom(6)
        b64_token = base64.urlsafe_b64encode(session_bytes).decode('utf-8').rstrip('=')[:8]

        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = max(0.95, target_height / 380.0)
        thickness = max(1, int(target_height / 500.0))
        line_height = int(target_height * 0.048)

        # REORGANIZED DISPLAY LAYOUT
        lines_data = [
            ("==================================================================================", (150, 150, 150)),
            ("                   INSIGHTFACE DUAL-AXIS IDENTITY & PERSONA REPORT", (0, 255, 255)),
            ("==================================================================================", (150, 150, 150)),
            ("[ SECTION A: VISUAL DEMOGRAPHICS & POSE ]", (255, 200, 100)),
            (f"  * TARGET METRICS  : {target_metrics_str} [3D Pitch: {pitch_t:+.1f}deg]", (200, 255, 200)),
            (f"  * SUSPECT METRICS : {suspect_metrics_str} [3D Pitch: {pitch_s:+.1f}deg]", (200, 255, 250)),
            ("----------------------------------------------------------------------------------", (100, 100, 100)),
            ("[ SECTION B: CORE BIOMETRIC MATCH EVALUATION ]", (255, 200, 100)),
            (f"  1. MASK MATCH (Face Mesh Cosine Dist) : {face_mesh_cosine_dist:.4f}   [{mask_likelihood}]", (0, 165, 255)),
            (f"  2. CANTHUS MATCH (Intercanthal Var)   : {canthal_var_pct:.2f}%     [{canthus_likelihood}]", (0, 255, 100) if canthus_match_pass else (0, 0, 255)),
            (f"  3. COMPOSITE SKELETAL INDEX VARIANCE  : {composite_skel_var:.2f}%     [{skeletal_likelihood}]", (0, 255, 200) if skeletal_index_pass else (0, 0, 255)),
            ("----------------------------------------------------------------------------------", (100, 100, 100)),
            ("[ SECTION C: SKELETAL INDEX ARRAY (Descending Weight Order) ]", (255, 255, 150))
        ]

        for item in skeletal_array:
            lines_data.append((
                f"     * [w={item['weight']:.2f}] {item['label']:<42}: {item['variance_pct']:.2f}% var ({item['invariance']})",
                (220, 220, 170)
            ))

        lines_data.extend([
            ("----------------------------------------------------------------------------------", (100, 100, 100)),
            ("[ SECTION D: SYNTHESIS & DUAL-HASH VERIFICATION ]", (255, 200, 100)),
            (f"  4. PERSONA ANALYTIC       : {persona_state}", (255, 100, 255)),
            (f"  * DATA SHA-256 (Hex)      : {sha_data_hex}", (170, 170, 170)),
            (f"  * IMAGE SHA-256 (Hex)     : {sha_img_hex}", (170, 170, 170)),
            (f"  * B64 TOKENS (Data|Image) : [{sha_data_b64} | {sha_img_b64}]  Session: [{b64_token}]", (140, 210, 140)),
            ("==================================================================================", (150, 150, 150))
        ])

        # --- LOG TO CONSOLE TERMINAL ---
        print("\n" + "="*82)
        print("                   INSIGHTFACE DUAL-AXIS IDENTITY & PERSONA REPORT")
        print("="*82)
        print("[ SECTION A: VISUAL DEMOGRAPHICS & POSE ]")
        print(f"  * TARGET METRICS  : {target_metrics_str} [3D Pitch: {pitch_t:+.1f}°]")
        print(f"  * SUSPECT METRICS : {suspect_metrics_str} [3D Pitch: {pitch_s:+.1f}°]")
        print("-"*82)
        print("[ SECTION B: CORE BIOMETRIC MATCH EVALUATION ]")
        print(f"  1. MASK MATCH (Face Mesh Cosine Dist) : {face_mesh_cosine_dist:.4f} [{mask_likelihood}]")
        print(f"  2. CANTHUS MATCH (Intercanthal Var)   : {canthal_var_pct:.2f}% [{canthus_likelihood}]")
        print(f"  3. COMPOSITE SKELETAL INDEX VARIANCE  : {composite_skel_var:.2f}% [{skeletal_likelihood}]")
        print("-"*82)
        print("[ SECTION C: SKELETAL INDEX ARRAY (Descending Weight Order) ]")
        for item in skeletal_array:
            print(f"     * [w={item['weight']:.2f}] {item['label']:<42}: {item['variance_pct']:.2f}% var ({item['invariance']})")
        print("-"*82)
        print("[ SECTION D: SYNTHESIS & DUAL-HASH VERIFICATION ]")
        print(f"  4. PERSONA ANALYTIC       : {persona_state}")
        print(f"  * DATA SHA-256 (Hex)      : {sha_data_hex}")
        print(f"  * IMAGE SHA-256 (Hex)     : {sha_img_hex}")
        print(f"  * B64 TOKENS (Data|Image) : [{sha_data_b64} | {sha_img_b64}] | Session: [{b64_token}]")
        print("="*82 + "\n")

        # --- DRAW ON IMAGE CANVAS ---
        box_height = (len(lines_data) * line_height) + int(line_height * 1.5)
        console_box = np.zeros((box_height, ribbon_w, 3), dtype=np.uint8)

        y_cursor = int(line_height * 1.2)
        x_padding = int(ribbon_w * 0.012)
        max_printable_w = ribbon_w - (x_padding * 2)

        for text_str, color_bgr in lines_data:
            applied_scale = calc_font_scale(text_str, font, font_scale, thickness, max_printable_w)
            cv2.putText(console_box, text_str, (x_padding, y_cursor), font, applied_scale, color_bgr, thickness, cv2.LINE_AA)
            y_cursor += line_height

        final_output_image = np.vstack((composite_ribbon, console_box))

        base_t = os.path.splitext(os.path.basename(target_path))[0]
        base_s = os.path.splitext(os.path.basename(suspect_path))[0]
        output_filename = f"{base_t}_{base_s}_SHA-{sha_data_b64}_TK-{b64_token}.jpg"
        
        export_path = os.path.join(os.path.dirname(target_path), output_filename)
        cv2.imwrite(export_path, final_output_image)
        print(f"↳ Successfully exported report image to:\n  {export_path}\n")

    except Exception as e:
        print(f"\n[Execution Error]: {e}\n")
