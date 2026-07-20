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

# --- HUGGING FACE DEMOGRAPHICS IMPORTS ---
from transformers import pipeline
from PIL import Image

hf_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

FAIRFACE_LABELS = [
    "White person", "Black person", "Indian person", 
    "East Asian person", "Southeast Asian person", 
    "Middle Eastern person", "Latino or Hispanic person"
]

AGE_LABELS = [
    "a photo of a child", "a photo of a young youth", 
    "a photo of a middle-aged adult", "a photo of an elderly senior citizen"
]

def analyze_demographics_huggingface(image_path):
    """ Extracts face crop via InsightFace and runs CLIP zero-shot classification for Race and Age """
    try:
        img = cv2.imread(image_path)
        if img is None: return "Offline", "Offline"
        faces = app.get(img)
        if not faces: return "Offline", "Offline"
        
        bbox = faces[0].bbox.astype(int)
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
        
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0: return "Offline", "Offline"
        
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_crop_rgb)
        
        race_results = hf_classifier(pil_img, candidate_labels=FAIRFACE_LABELS)
        race_label = race_results[0]['label'].replace(" person", "").title()
        race_conf = race_results[0]['score'] * 100
        race_str = f"{race_label} ({race_conf:.1f}%)"
        
        age_results = hf_classifier(pil_img, candidate_labels=AGE_LABELS)
        top_age = age_results[0]['label']
        age_map = {
            "a photo of a child": "Child (0-12)",
            "a photo of a young youth": "Youth (13-29)",
            "a photo of a middle-aged adult": "Adult (30-59)",
            "a photo of an elderly senior citizen": "Senior (60+)"
        }
        age_str = age_map.get(top_age, "Adult (30-59)")
        
        return race_str, age_str
    except Exception:
        return "Offline", "Offline"

def get_pupil_centers(landmarks):
    """ Computes rough pupil centers based on standard 106-point landmark indices. """
    if len(landmarks) >= 96:
        left_pupil = np.mean(landmarks[35:41], axis=0)
        right_pupil = np.mean(landmarks[89:95], axis=0)
    else:
        left_pupil = np.mean(landmarks[0:4], axis=0)
        right_pupil = np.mean(landmarks[4:8], axis=0)
    return left_pupil, right_pupil

def process_face_data(image_path):
    """ Detects faces, extracts the original identity embedding, and gets points """
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
        
    return img, landmarks, face.normed_embedding

def generate_pupil_aligned_mesh(img, landmarks, target_w, target_h, color, label_text):
    """ Normalizes, scales, and aligns a face mesh with a localized overlay header string. """
    left_pupil, right_pupil = get_pupil_centers(landmarks)
    
    pupil_center = (left_pupil + right_pupil) / 2.0
    pupil_distance = np.linalg.norm(left_pupil - right_pupil)
    
    desired_dist = target_w * 0.25
    scale = desired_dist / pupil_distance
    
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas_center = np.array([target_w / 2.0, target_h / 2.0])
    
    aligned_landmarks = (landmarks - pupil_center) * scale + canvas_center
    
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

    # Render target/suspect structural title alignment labels inside mesh frames
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, target_h / 800.0)
    thickness = max(1, int(target_h / 450.0))
    text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
    
    x_pos = int((target_w - text_size[0]) / 2)
    y_pos = int(target_h * 0.06)
    
    # Text background strip drop-shadow for clarity
    cv2.putText(canvas, label_text, (x_pos, y_pos), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, label_text, (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
            
    return canvas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InsightFace Base64 Integrated Verification Pipeline")
    parser.add_argument("target", help="Filename or path of target image")
    parser.add_argument("suspect", help="Filename or path of suspect image")
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    IMG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "img"))

    target_path = os.path.abspath(args.target) if os.path.exists(args.target) else os.path.join(IMG_DIR, args.target)
    suspect_path = os.path.abspath(args.suspect) if os.path.exists(args.suspect) else os.path.join(IMG_DIR, args.suspect)

    if not os.path.exists(target_path) or not os.path.exists(suspect_path):
        print(f"[Error]: Source files missing.")
        sys.exit(1)

    try:
        print("\n--- Running Stable Identity Framework ---")
        
        t_img, t_landmarks, vector_target = process_face_data(target_path)
        s_img, s_landmarks, vector_suspect = process_face_data(suspect_path)

        cosine_dist = distance.cosine(vector_target, vector_suspect)
        euclidean_dist = distance.euclidean(vector_target, vector_suspect)

        h1, w1 = t_img.shape[:2]
        h2, w2 = s_img.shape[:2]
        target_height = min(h1, h2)
        
        t_scale, s_scale = target_height / h1, target_height / h2
        t_img_res = cv2.resize(t_img, (int(w1 * t_scale), target_height))
        s_img_res = cv2.resize(s_img, (int(w2 * s_scale), target_height))

        color_t = (255, 140, 0)   # Deep Orange
        color_s = (30, 144, 255)  # Dodger Blue
        
        # Inject context text into the upper layout profiles
        t_mesh_canvas = generate_pupil_aligned_mesh(t_img, t_landmarks, t_img_res.shape[1], target_height, color_t, "TARGET FACE MESH")
        s_mesh_canvas = generate_pupil_aligned_mesh(s_img, s_landmarks, s_img_res.shape[1], target_height, color_s, "SUSPECT FACE MESH")

        composite_ribbon = np.hstack((t_img_res, t_mesh_canvas, s_img_res, s_mesh_canvas))
        ribbon_w = composite_ribbon.shape[1]

        target_race, target_age = analyze_demographics_huggingface(target_path)
        suspect_race, suspect_age = analyze_demographics_huggingface(suspect_path)

        if cosine_dist < 0.25:
            forensic_verdict = "MATCH: Absolute identity match (Preserved features / same era)."
        elif 0.25 <= cosine_dist < 0.35:
            forensic_verdict = "MATCH: Confident identity match (Typical real-world variations)."
        elif 0.35 <= cosine_dist < 0.45:
            forensic_verdict = "CRITICAL RISK: High-fidelity Doppelganger / Lookalike isolated."
        elif 0.45 <= cosine_dist <= 0.55:
            forensic_verdict = "AMBIGUOUS ZONE: Structural similarities detected."
        elif 0.55 < cosine_dist <= 0.68:
            forensic_verdict = "HISTORICAL/GAP ZONE: Probable Cross-Age Match or verified lookalike."
        else:
            forensic_verdict = "NO MATCH: Distinct structures isolated."

        # --- ADVANCED BASE64 ENCODING SYSTEMS ---
        raw_report_string = f"{target_race}{suspect_race}{cosine_dist:.4f}"
        sha256_raw = hashlib.sha256(raw_report_string.encode('utf-8')).digest()
        
        # Convert SHA256 bytes directly to URL-safe Base64 and clean tracking padding
        b64_hash_short = base64.urlsafe_b64encode(sha256_raw).decode('utf-8').replace('=', '').replace('_', '').replace('-', '')[:12]
        
        # Generate true uniform random bytes for a crisp 8-character Base64 token string
        session_bytes = os.urandom(6)
        b64_token = base64.urlsafe_b64encode(session_bytes).decode('utf-8').replace('=', '')[:8]

        # Formulate metrics box layout
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = max(1.1, target_height / 320.0)
        thickness = max(1, int(target_height / 420.0))
        line_height = int(target_height * 0.070)
        sha_hex_string = hashlib.sha256(raw_report_string.encode('utf-8')).hexdigest()

        lines_data = [
            ("=========================================================================", (150, 150, 150)),
            ("               INSIGHTFACE BIOMETRIC IDENTITY & ALIGNMENT REPORT", (0, 255, 255)),
            ("=========================================================================", (150, 150, 150)),
            (f"TARGET HF METRICS:  Race: {target_race} | Age Range: {target_age}", (200, 255, 200)),
            (f"SUSPECT HF METRICS: Race: {suspect_race} | Age Range: {suspect_age}", (200, 255, 250)),
            ("-------------------------------------------------------------------------", (100, 100, 100)),
            (f"COSINE DISTANCE:    {cosine_dist:.4f}  (Match Baseline Threshold < 0.35)", (0, 165, 255)),
            (f"EUCLIDEAN DISTANCE: {euclidean_dist:.4f}", (255, 180, 70)),
            (f"FORENSIC ANALYTIC:  {forensic_verdict}", (255, 100, 255)),
            ("-------------------------------------------------------------------------", (100, 100, 100)),
            (f"SHA-256 Checksum (Hex):    {sha_hex_string[:32]}...", (170, 170, 170)),  
            (f"SHA-256 Compress (Base64): {b64_hash_short}", (140, 210, 140)),
            (f"Base64 Session Token:      [{b64_token}]", (255, 255, 0)),
            ("=========================================================================", (150, 150, 150))
        ]

        box_height = (len(lines_data) * line_height) + int(line_height * 1.5)
        console_box = np.zeros((box_height, ribbon_w, 3), dtype=np.uint8)

        y_cursor = int(line_height * 1.2)
        x_padding = int(ribbon_w * 0.012)
        
        for text_str, color_bgr in lines_data:
            cv2.putText(console_box, text_str, (x_padding, y_cursor), font, font_scale, color_bgr, thickness, cv2.LINE_AA)
            y_cursor += line_height

        final_output_image = np.vstack((composite_ribbon, console_box))

        # Build clean cryptographic profile name strings
        base_t = os.path.splitext(os.path.basename(target_path))[0]
        base_s = os.path.splitext(os.path.basename(suspect_path))[0]
        output_filename = f"{base_t}_{base_s}_SHA-{b64_hash_short}_TK-{b64_token}.jpg"
        
        export_path = os.path.join(os.path.dirname(target_path), output_filename)
        cv2.imwrite(export_path, final_output_image)
        print(f"↳ Successfully exported unified biometric report to:\n  {export_path}\n")

    except Exception as e:
        print(f"\n[Execution Error]: {e}\n")
