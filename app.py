import cv2
import mediapipe as mp
import numpy as np
import os
import math
import time
import threading
import requests
import base64
from dotenv import load_dotenv

# Load environment variables relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path, override=True)

HF_KEY = os.getenv("HF_KEY")
HF_MODEL = os.getenv("HF_MODEL", "runwayml/stable-diffusion-v1-5")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# Debug print to verify load status (masked for security)
if HF_KEY:
    masked_key = HF_KEY[:10] + "..." if len(HF_KEY) > 10 else HF_KEY
    print(f"[Debug] Loaded HF_KEY: {masked_key} (length: {len(HF_KEY)})")
else:
    print("[Debug] Loaded HF_KEY: None")

# Determine Mode
mode = "LOCAL_MOCK"
if HF_KEY and "your_hf_token" not in HF_KEY and HF_KEY != "":
    mode = "HF_API"
    print(f"[System] Hugging Face Token detected. Running in SD 1.5 API Mode using model: {HF_MODEL}")
else:
    print("[System] No Hugging Face Token found in .env. Running in LOCAL MOCK MODE.")

is_api_mode = (mode != "LOCAL_MOCK")

# Define style presets
STYLES = [
    {
        "name": "Anime", 
        "prompt": "masterpiece anime style illustration, breathtaking scenery, detailed anime character, makoto shinkai aesthetic, gorgeous lighting, vibrant colors, co mix wave films, key visual, 8k resolution, highly detailed",
        "mock_name": "Makoto Shinkai"
    },
    {
        "name": "Pixar 3D", 
        "prompt": "gorgeous 3d pixar disney animation style, highly detailed character design, warm volumetric lighting, raytracing, octane render, stylized 3d art, masterpiece",
        "mock_name": "Clay Bloom"
    },
    {
        "name": "Chibi", 
        "prompt": "ultra cute chibi 3d character style, glossy claymation, large expressive sparkling eyes, adorable, soft pastel colors, cinematic lighting, toy design, high quality, masterpiece",
        "mock_name": "Pastel Watercolor"
    },
    {
        "name": "Manga", 
        "prompt": "masterpiece manga page illustration, detailed black and white ink sketch, professional line art, crosshatching, dramatic screentone, dynamic action pose, clean comic book drawing",
        "mock_name": "Ink Halftone"
    },
    {
        "name": "Cyberpunk", 
        "prompt": "cyberpunk aesthetic, glowing neon lights, futuristic cityscape, dramatic synthwave color scheme, high contrast reflections, moody wet streets, blade runner style, high tech cybernetics, masterpiece",
        "mock_name": "Neon Glow"
    },
    {
        "name": "Oil Painting", 
        "prompt": "masterpiece impressionist oil painting style, rich thick textured brushstrokes, van gogh starry night aesthetic, impasto canvas texture, vibrant swirling colors, dramatic warm lighting",
        "mock_name": "Vibrant Impasto"
    },
    {
        "name": "Pixel Art", 
        "prompt": "masterpiece 16-bit pixel art style, highly detailed retro video game landscape and character, vibrant color palette, nostalgic arcade aesthetic, clean pixels, high quality",
        "mock_name": "16-Bit Grid"
    },
    {
        "name": "Watercolor", 
        "prompt": "exquisite watercolor illustration style, soft flowing pastel colors, artistic splatters, fine detailed ink outlines, textured watercolor paper, beautiful organic look, masterpiece",
        "mock_name": "Fluid Sketch"
    },
    {
        "name": "Blueprint", 
        "prompt": "detailed technical blueprint schematics style, clean white line art drawing on dark blue grid background, engineering draft, precise architectural lines, high quality, masterpiece",
        "mock_name": "Technical Grid"
    },
    {
        "name": "Pop Art", 
        "prompt": "andy warhol pop art style, bold retro silkscreen print, high contrast block colors, saturated neon color palette, iconic vintage pop culture aesthetic, masterpiece",
        "mock_name": "Warhol 4-Grid"
    }
]

# State Variables
current_style_idx = 0
styled_image = None
is_processing = False
last_request_time = 0
MIN_INTERVAL = 3.0  # Safe interval between API calls (seconds)
touch_active = False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,  # Use Lite model for fast CPU tracking
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# Helper: Distance calculation
def get_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Local Mock Filters
def apply_mock_filter(crop_img, style_name):
    h, w = crop_img.shape[:2]
    
    if style_name == "Anime":
        # Anime Style: Bilateral smoothing + warm glowing colors + detail enhancement
        small = cv2.resize(crop_img, (256, 256))
        smoothed = cv2.bilateralFilter(small, d=9, sigmaColor=40, sigmaSpace=40)
        
        # Color processing: Make it warm and vibrant (Makoto Shinkai feel)
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255) # Saturated
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Brighter
        color_boost = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Apply detail enhancement for an illustrative painted look
        enhanced = cv2.detailEnhance(color_boost, sigma_s=10, sigma_r=0.15)
        
        # Combine with subtle outlines
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Blend edges and color
        anime = cv2.bitwise_and(enhanced, edges_color)
        anime = cv2.addWeighted(anime, 0.85, enhanced, 0.15, 0)
        return cv2.resize(anime, (w, h), interpolation=cv2.INTER_LINEAR)
        
    elif style_name == "Pixar 3D":
        # Clay render + soft bloom/glow
        small = cv2.resize(crop_img, (256, 256))
        clay = cv2.bilateralFilter(small, d=13, sigmaColor=80, sigmaSpace=80)
        
        # Color processing
        hsv = cv2.cvtColor(clay, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
        clay_colored = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Bloom effect (soft overlay of bright areas)
        blur_glow = cv2.GaussianBlur(clay_colored, (21, 21), 0)
        bloom = cv2.addWeighted(clay_colored, 0.75, blur_glow, 0.25, 10)
        return cv2.resize(bloom, (w, h), interpolation=cv2.INTER_CUBIC)
        
    elif style_name == "Chibi":
        # Pastel watercolor cute portrait
        small = cv2.resize(crop_img, (256, 256))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.85, 0, 255) # Soft pastel
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 40, 0, 255)   # High-key brightness
        pastel = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Stylize
        chibi = cv2.stylization(pastel, sigma_s=50, sigma_r=0.2)
        
        # Pink blush vignette
        mask = np.zeros_like(chibi)
        cv2.circle(mask, (128, 128), 160, (230, 190, 255), -1)
        chibi_pink = cv2.addWeighted(chibi, 0.85, mask, 0.15, 0)
        return cv2.resize(chibi_pink, (w, h), interpolation=cv2.INTER_LINEAR)
        
    elif style_name == "Manga":
        # Black and white pencil sketch + ink thresholding
        small = cv2.resize(crop_img, (256, 256))
        sketch_gray, _ = cv2.pencilSketch(small, sigma_s=50, sigma_r=0.07, shade_factor=0.03)
        sketch_gray = cv2.equalizeHist(sketch_gray)
        _, thresh = cv2.threshold(sketch_gray, 180, 255, cv2.THRESH_BINARY)
        manga = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return cv2.resize(manga, (w, h), interpolation=cv2.INTER_NEAREST)
        
    elif style_name == "Cyberpunk":
        # Neon edge glowing on dark cool background
        small = cv2.resize(crop_img, (256, 256))
        dark = small.astype(np.float32)
        dark[:, :, 0] = np.clip(dark[:, :, 0] * 1.5 + 20, 0, 255) # Blue channel boost
        dark[:, :, 1] = np.clip(dark[:, :, 1] * 0.3, 0, 255)
        dark[:, :, 2] = np.clip(dark[:, :, 2] * 0.8, 0, 255)
        dark = dark.astype(np.uint8)
        
        # Detect edges
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        
        # Glow edge layer
        neon = np.zeros_like(small)
        neon[edges > 0] = [238, 0, 238] # Neon Magenta BGR
        neon_glow = cv2.GaussianBlur(neon, (11, 11), 0)
        
        # Combine
        cyber = cv2.addWeighted(dark, 0.7, neon_glow, 0.5, 0)
        cyber[edges > 0] = [238, 238, 0] # Cyan edges BGR
        return cv2.resize(cyber, (w, h), interpolation=cv2.INTER_LINEAR)
        
    elif style_name == "Oil Painting":
        # Stylized brush strokes + vivid color boost
        oil = cv2.stylization(crop_img, sigma_s=60, sigma_r=0.45)
        hsv = cv2.cvtColor(oil, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
    elif style_name == "Pixel Art":
        # 16-Bit game look
        grid_w, grid_h = 64, 64
        small = cv2.resize(crop_img, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
        quantized = (small // 48) * 48 + 24
        hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
        quantized_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.resize(quantized_color, (w, h), interpolation=cv2.INTER_NEAREST)
        
    elif style_name == "Watercolor":
        # Stylized fluid color with delicate pencil outlines
        watercolor = cv2.stylization(crop_img, sigma_s=40, sigma_r=0.15)
        small = cv2.resize(crop_img, (256, 256))
        sketch_gray, _ = cv2.pencilSketch(small, sigma_s=30, sigma_r=0.07, shade_factor=0.03)
        sketch_color = cv2.resize(cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR), (w, h))
        return cv2.addWeighted(watercolor, 0.85, sketch_color, 0.15, 0)
        
    elif style_name == "Blueprint":
        # Navy blue blueprint with white lines and grid
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 110)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        
        bg = np.zeros_like(crop_img)
        bg[:] = [100, 30, 5]
        
        # Grid lines
        grid_space = 25
        for x in range(0, w, grid_space):
            cv2.line(bg, (x, 0), (x, h), (130, 45, 10), 1)
        for y in range(0, h, grid_space):
            cv2.line(bg, (0, y), (w, y), (130, 45, 10), 1)
            
        bg[edges > 0] = [255, 255, 230]
        return bg
        
    elif style_name == "Pop Art":
        # Andy Warhol 4-quadrant screen print
        small = cv2.resize(crop_img, (128, 128))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        q1 = np.zeros((128, 128, 3), dtype=np.uint8)
        q1[:] = [180, 105, 255] # Pink
        q1[thresh > 0] = [255, 255, 0] # Cyan
        
        q2 = np.zeros((128, 128, 3), dtype=np.uint8)
        q2[:] = [0, 0, 255] # Red
        q2[thresh > 0] = [0, 255, 255] # Yellow
        
        q3 = np.zeros((128, 128, 3), dtype=np.uint8)
        q3[:] = [255, 0, 0] # Blue
        q3[thresh > 0] = [0, 128, 255] # Orange
        
        q4 = np.zeros((128, 128, 3), dtype=np.uint8)
        q4[:] = [0, 255, 0] # Green
        q4[thresh > 0] = [128, 0, 128] # Purple
        
        top = np.hstack((q1, q2))
        bottom = np.hstack((q3, q4))
        grid = np.vstack((top, bottom))
        return cv2.resize(grid, (w, h))

    return crop_img

# API Client background thread worker
def api_thread_worker(crop_img, prompt_text, style_idx):
    global styled_image, is_processing, is_api_mode, mode
    try:
        if mode == "HF_API":
            print(f"[API] Starting generation on Hugging Face for style: {STYLES[style_idx]['name']}...")
            _, buffer = cv2.imencode('.jpg', crop_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            api_url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
            headers = {
                "Authorization": f"Bearer {HF_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": img_base64,
                "parameters": {
                    "prompt": prompt_text,
                    "strength": 1.0,
                    "num_inference_steps": 30,
                    "guidance_scale": 10.0
                }
            }
            
            resp = requests.post(api_url, json=payload, headers=headers)
            
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "json" in content_type:
                    try:
                        err_json = resp.json()
                        print(f"[API] Hugging Face returned JSON: {err_json}")
                        if "estimated_time" in err_json:
                            print(f"[API] Model is loading. Estimated time: {err_json['estimated_time']:.1f}s. Please wait...")
                    except Exception as json_err:
                        print(f"[API] Error parsing HF JSON: {json_err}")
                else:
                    nparr = np.frombuffer(resp.content, np.uint8)
                    decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if decoded_img is not None:
                        styled_image = cv2.resize(decoded_img, (512, 512))
                        print(f"[API] Success! Hugging Face styled image retrieved.")
                    else:
                        print(f"[API] Failed to decode image from Hugging Face response.")
            elif resp.status_code == 503:
                try:
                    err_json = resp.json()
                    est = err_json.get("estimated_time", 20)
                    print(f"[API] Hugging Face model is loading (503). Estimated time: {est:.1f}s. Please wait...")
                except:
                    print(f"[API] Hugging Face model is loading (503). Please wait...")
            else:
                print(f"[API] Hugging Face returned status code {resp.status_code}: {resp.text}")
                if resp.status_code in [400, 403, 404]:
                    is_api_mode = False
                    mode = "LOCAL_MOCK"
                    print("\n[System] Permanent API issue detected (Hugging Face has disabled free Image-to-Image models on this serverless tier).")
                    print("[System] Automatically falling back to LOCAL MOCK mode with premium, gorgeous artistic filters!\n")
                
    except Exception as e:
        print(f"[API] Exception occurred: {e}")
        print("[API] Hint: Make sure HF_KEY in .env is valid.")
    finally:
        is_processing = False

# Start webcam capture
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"[Error] Could not open webcam with index {CAMERA_INDEX}.")
    exit()

# Set camera resolution to 640x480 for smooth tracking and processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n--- Hand Frame Magic Lens Started ---")
print("Instructions:")
print("1. Place both hands in webcam field.")
print("2. Form a frame with index and thumb tips of both hands.")
print("3. TOUCH left index+thumb AND right index+thumb concurrently to switch styles.")
print("4. Press 'q' in the window to quit.")
print("--------------------------------------\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip horizontally for mirrored/natural feedback
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Create a clean copy of the frame for cropping BEFORE drawing outlines
    frame_clean = frame.copy()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Track hands
    results = hands.process(frame_rgb)
    
    # Store fingertips coordinates
    fingertips = []
    hand_scales = []
    touch_statuses = []
    
    if results.multi_hand_landmarks:
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            # Draw standard landmarks on hands for visual feedback
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks in pixel values
            lms = []
            for lm in hand_lms.landmark:
                lms.append((int(lm.x * w), int(lm.y * h)))
            
            # Extract thumb tip (4) and index tip (8)
            thumb_tip = lms[4]
            index_tip = lms[8]
            
            # Add to coordinate list
            fingertips.append(thumb_tip)
            fingertips.append(index_tip)
            
            # Calculate scale of the hand (Wrist [0] to Middle Finger MCP [9])
            scale = get_distance(lms[0], lms[9])
            if scale == 0:
                scale = 1.0
            
            # Distance between index and thumb tip
            touch_dist = get_distance(thumb_tip, index_tip)
            ratio = touch_dist / scale
            hand_scales.append(scale)
            
            # Determine if this hand is touching (ratio < 0.35)
            is_touching = ratio < 0.35
            touch_statuses.append(is_touching)
            
            # Label on the hand showing ratio
            cv2.putText(
                frame, 
                f"Dist: {ratio:.2f}", 
                (thumb_tip[0], thumb_tip[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                (0, 255, 0) if not is_touching else (0, 0, 255), 
                1
            )

    # We need exactly 2 hands in view to form the 4-point box
    has_four_points = len(fingertips) == 4
    
    # Touch state tracking
    if len(touch_statuses) == 2:
        both_touching = all(touch_statuses)
        if both_touching and not touch_active:
            touch_active = True
            print("[System] Hands closed! Transitioning style...")
        elif not both_touching and touch_active:
            # Trigger style change when hands open again
            touch_active = False
            current_style_idx = (current_style_idx + 1) % len(STYLES)
            styled_image = None  # Clear previous style image
            print(f"[System] Hands opened! Switched style to: {STYLES[current_style_idx]['name']}")

    # Render hand box and apply style if we have 4 tips
    if has_four_points:
        # Map 4 points to top-left, top-right, bottom-right, bottom-left
        # Sort by vertical coordinates (Y) to separate top two and bottom two
        top_pts = sorted(fingertips, key=lambda p: p[1])[:2]
        bottom_pts = sorted(fingertips, key=lambda p: p[1])[2:]
        
        # Sort top two by X to get top-left and top-right
        TL = min(top_pts, key=lambda p: p[0])
        TR = max(top_pts, key=lambda p: p[0])
        
        # Sort bottom two by X to get bottom-left and bottom-right
        BL = min(bottom_pts, key=lambda p: p[0])
        BR = max(bottom_pts, key=lambda p: p[0])
        
        # Define source points array (ensure floats)
        src_pts = np.array([TL, TR, BR, BL], dtype=np.float32)
        dst_pts = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], dtype=np.float32)
        
        # Draw frame boundary
        box_color = (0, 0, 255) if touch_active else (255, 255, 0) # Red if closed, Cyan/Yellow if open
        cv2.polylines(frame, [np.int32(src_pts)], isClosed=True, color=box_color, thickness=3)
        
        # Draw circles at vertices
        for pt in [TL, TR, BR, BL]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 255, 255), -1)

        # Extraction and styling loop
        if not touch_active:
            # Get perspective transform matrix and crop image inside box
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            crop_img = cv2.warpPerspective(frame_clean, M, (512, 512))
            
            # API Mode handling
            if is_api_mode:
                # Trigger styling if not busy and minimum interval has passed
                current_time = time.time()
                if not is_processing and (current_time - last_request_time > MIN_INTERVAL):
                    is_processing = True
                    last_request_time = current_time
                    
                    # Spawn threading task to prevent OpenCV display freeze
                    prompt = STYLES[current_style_idx]["prompt"]
                    thread = threading.Thread(
                        target=api_thread_worker, 
                        args=(crop_img.copy(), prompt, current_style_idx)
                    )
                    thread.daemon = True
                    thread.start()
            else:
                # Local Mock Mode: apply instant filter
                mock_styled = apply_mock_filter(crop_img, STYLES[current_style_idx]["name"])
                styled_image = mock_styled
            
            # Draw styled image back onto original frame using inverse warp
            if styled_image is not None:
                M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
                warped_styled = cv2.warpPerspective(styled_image, M_inv, (w, h))
                
                # Apply mask to draw only inside quadrilateral
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.int32(src_pts), 255)
                
                # Overlay
                frame[mask > 0] = warped_styled[mask > 0]
                
                # Draw small status text near frame
                status_txt = f"{STYLES[current_style_idx]['name']}"
                if not is_api_mode:
                    status_txt += f" ({STYLES[current_style_idx]['mock_name']})"
                cv2.putText(
                    frame, status_txt, 
                    (int(TL[0]), int(TL[1]) - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )
    
    # Status Board UI overlays
    style_info = STYLES[current_style_idx]
    if mode == "HF_API":
        mode_str = "SD 1.5 API (Hugging Face)"
    else:
        mode_str = "LOCAL MOCK Mode (Offline)"
    
    # Header display
    cv2.rectangle(frame, (10, 10), (450, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"MODE: {mode_str}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"STYLE: {style_info['name']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show loading status in API mode
    if is_api_mode and is_processing:
        cv2.putText(frame, "STYLING... [SD]", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    elif is_api_mode:
        cv2.putText(frame, "READY [Idle]", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"FILTER: {style_info['mock_name']}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Show window
    cv2.imshow("Hand Frame Magic Lens", frame)

    # Listen for keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'): # Manual override style swap key
        current_style_idx = (current_style_idx + 1) % len(STYLES)
        styled_image = None
        print(f"[Manual] Switched style to: {STYLES[current_style_idx]['name']}")

# Clean up
cap.release()
cv2.destroyAllWindows()
hands.close()
print("[System] Application terminated.")
