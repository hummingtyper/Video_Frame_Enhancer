import streamlit as st
import cv2
import numpy as np
import tempfile
import io
import os
from pathlib import Path
from typing import Tuple

# --- 1. PERSISTENCE HELPERS ---
def get_query_param(key, default, cast_func=str):
    if key in st.query_params:
        val = st.query_params[key]
        try:
            return cast_func(val)
        except:
            return default
    return default

def update_query_param(key):
    val = st.session_state[key]
    st.query_params[key] = str(val)

to_bool = lambda x: str(x).lower() == 'true'
to_int = int
to_float = float

# --- TRANSLATION CONFIGURATION ---
TRANSLATIONS = {
    "English": {
        "title": "🎞️ Advanced Retro Video Enhancer",
        "upload_label": "Choose a video file",
        "sidebar_settings": "Enhancement Settings",
        "lang_select": "Language",
        "frame_idx": "Select Frame",
        "timestamp": "Time (s)",
        "win_size": "Window Size (Temporal Denoising)",
        "merge_mode": "Optimization Mode",
        "sharpen": "Sharpen Amount",
        "use_clahe": "Use CLAHE (Contrast)",
        "clahe_clip": "CLAHE Clip Limit",
        "sigma_factor": "Sigma Factor (Outlier Threshold)",
        "chk_enhance": "Enable Live Enhancement",
        "header_orig": "Original (Preview)",
        "header_enh": "Enhanced",
        "dl_orig": "Download Original",
        "dl_enh": "Download Enhanced",
        "err_load": "Error loading video",
        "err_process": "Could not process frame",
        "init_spinner": "Initializing Video Engine...",
        "proc_spinner": "Processing... (Aligning & Merging)",
        "prev": "◀ Previous",
        "next": "Next ▶",
        "enhance_tip": "Check this to automatically enhance frames as you browse.",
        "vis_label": "Window Visualization",
        "mode_desc": {
            "mean": "Standard Average (Reduces noise efficiently)",
            "median": "Median (Removes moving objects/ghosting)",
            "gaussian": "Gaussian (Center-weighted, reduces ghosting)",
            "sigma_clip": "Gaussian-Sigma (Smart: Removes scratches + Smooths motion)",
            "min": "Minimum (Removes bright flashes/dust)",
            "max": "Maximum (Removes dark spots)"
        }
    },
    "Deutsch": {
        "title": "🎞️ Retro Video Optimierer",
        "upload_label": "Videodatei auswählen",
        "sidebar_settings": "Optimierungs-Einstellungen",
        "lang_select": "Sprache",
        "frame_idx": "Frame auswählen",
        "timestamp": "Zeit (s)",
        "win_size": "Fenstergrösse (Zeitliches Rauschen)",
        "merge_mode": "Optimierungsmodus",
        "sharpen": "Schärfe-Intensität",
        "use_clahe": "CLAHE nutzen (Kontrast)",
        "clahe_clip": "CLAHE Limit",
        "sigma_factor": "Sigma Faktor (Toleranzgrenze)",
        "chk_enhance": "Live-Optimierung aktivieren",
        "header_orig": "Original (Vorschau)",
        "header_enh": "Optimiert",
        "dl_orig": "Original herunterladen",
        "dl_enh": "Optimiertes Bild herunterladen",
        "err_load": "Fehler beim Laden des Videos",
        "err_process": "Frame konnte nicht verarbeitet werden",
        "init_spinner": "Video-Engine wird gestartet...",
        "proc_spinner": "Verarbeite... (Ausrichtung & Denoising)",
        "prev": "◀ Zurück",
        "next": "Vor ▶",
        "enhance_tip": "Aktivieren, um Frames beim Durchblättern automatisch zu optimieren.",
        "vis_label": "Fenster-Visualisierung",
        "mode_desc": {
            "mean": "Mittelwert (Standard Rauschreduzierung)",
            "median": "Median (Entfernt bewegte Objekte)",
            "gaussian": "Gauss (Zentrumsgewichtet, weniger Geisterbilder)",
            "sigma_clip": "Gauss-Sigma (Smart: Entfernt Kratzer + Glättet)",
            "min": "Minimum (Entfernt helle Blitzer/Staub)",
            "max": "Maximum (Entfernt dunkle Flecken)"
        }
    }
}

# --- ENGINE CLASS ---
class AdvancedRetroEnhancer:
    def __init__(self, input_video_path: str):
        self.input_path = Path(input_video_path)
        self.cap = cv2.VideoCapture(str(self.input_path))
        
        if not self.cap.isOpened():
            raise IOError(f"Video error: {self.input_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.warp_mode = cv2.MOTION_AFFINE
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        
        self._cache_frame_idx = -1
        self._cache_window_size = -1
        self._cache_merge_mode = ""
        self._cache_sigma = -1
        self._cached_merged_image = None 

    def get_raw_frame(self, idx):
        idx = np.clip(idx, 0, self.total_frames - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def get_stack_thumbnails(self, center_idx, window_size, thumb_height=64):
        """Fetch small thumbnails for visualization."""
        padding = window_size // 2
        thumbnails = []
        captions = []
        
        for offset in range(-padding, padding + 1):
            read_idx = np.clip(center_idx + offset, 0, self.total_frames - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, read_idx)
            ret, frame = self.cap.read()
            
            if ret:
                # Resize for performance
                h, w = frame.shape[:2]
                scale = thumb_height / float(h)
                new_w = int(w * scale)
                thumb = cv2.resize(frame, (new_w, thumb_height))
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                
                # Add red border to CENTER frame
                if offset == 0:
                    thumb = cv2.copyMakeBorder(thumb, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 0, 0])
                    captions.append("**Center**")
                else:
                    captions.append(f"{offset:+d}")
                    
                thumbnails.append(thumb)
            else:
                thumbnails.append(np.zeros((thumb_height, thumb_height, 3), dtype=np.uint8))
                captions.append("N/A")
                
        return thumbnails, captions

    def _get_frame_stack(self, center_idx: int, window_size: int) -> list:
        padding = window_size // 2
        stack = []
        for offset in range(-padding, padding + 1):
            read_idx = np.clip(center_idx + offset, 0, self.total_frames - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, read_idx)
            ret, frame = self.cap.read()
            if ret:
                stack.append(frame)
        return stack

    def _align_images(self, target, stack):
        aligned_stack = []
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        for img in stack:
            if img is target: 
                aligned_stack.append(img)
                continue
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                _, warp_matrix = cv2.findTransformECC(
                    target_gray, img_gray, warp_matrix, self.warp_mode, self.criteria
                )
                h, w = target.shape[:2]
                aligned = cv2.warpAffine(
                    img, warp_matrix, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                )
                aligned_stack.append(aligned)
            except cv2.error:
                aligned_stack.append(img)
        return aligned_stack

    def _apply_clahe(self, img_bgr, clip_limit=2.0, grid_size=8):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        l_enhanced = clahe.apply(l)
        lab_merged = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    def _apply_sharpen(self, img_bgr, amount=1.0):
        if amount <= 0: return img_bgr
        gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 3.0)
        sharpened = cv2.addWeighted(img_bgr, 1.0 + amount, gaussian, -amount, 0)
        return sharpened

    def process_pipeline(self, frame_idx, window_size, merge_mode, sharpen_amount, use_clahe, clahe_clip, sigma_factor=2.0):
        calc_heavy_lifting = True
        
        if (frame_idx == self._cache_frame_idx and 
            window_size == self._cache_window_size and 
            merge_mode == self._cache_merge_mode and
            sigma_factor == self._cache_sigma and
            self._cached_merged_image is not None):
            calc_heavy_lifting = False
        
        if calc_heavy_lifting:
            stack = self._get_frame_stack(frame_idx, window_size)
            if not stack: return None, None
            target_frame = stack[window_size // 2]
            
            if window_size > 1:
                aligned_stack = self._align_images(target_frame, stack)
                stack_arr = np.array(aligned_stack, dtype=np.float32)
                
                # --- MERGE LOGIC ---
                if merge_mode == 'median':
                    merged = np.median(stack_arr, axis=0)
                elif merge_mode == 'min':
                    merged = np.min(stack_arr, axis=0)
                elif merge_mode == 'max':
                    merged = np.max(stack_arr, axis=0)
                elif merge_mode == 'gaussian':
                    sigma = window_size / 2.5
                    x = np.linspace(-(window_size//2), window_size//2, window_size)
                    kernel = np.exp(-0.5 * (x / sigma)**2)
                    kernel /= kernel.sum()
                    weights = kernel.reshape(window_size, 1, 1, 1)
                    merged = np.sum(stack_arr * weights, axis=0)
                elif merge_mode == 'sigma_clip':
                    mean_val = np.mean(stack_arr, axis=0)
                    std_val = np.std(stack_arr, axis=0)
                    lower = mean_val - (sigma_factor * std_val)
                    upper = mean_val + (sigma_factor * std_val)
                    mask = (stack_arr >= lower) & (stack_arr <= upper)
                    sigma_gauss = window_size / 2.5
                    x = np.linspace(-(window_size//2), window_size//2, window_size)
                    kernel = np.exp(-0.5 * (x / sigma_gauss)**2)
                    weights = kernel.reshape(window_size, 1, 1, 1)
                    weighted_stack = stack_arr * weights * mask
                    weight_sum = np.sum(weights * mask, axis=0)
                    weight_sum[weight_sum == 0] = 1.0 
                    merged = np.sum(weighted_stack, axis=0) / weight_sum
                else: 
                    merged = np.mean(stack_arr, axis=0)
                # --- END MERGE LOGIC ---

                self._cached_merged_image = np.clip(merged, 0, 255).astype(np.uint8)
            else:
                self._cached_merged_image = target_frame

            self._cache_frame_idx = frame_idx
            self._cache_window_size = window_size
            self._cache_merge_mode = merge_mode
            self._cache_sigma = sigma_factor
            self._original_for_display = target_frame

        result = self._cached_merged_image.copy()
        if sharpen_amount > 0:
            result = self._apply_sharpen(result, amount=sharpen_amount)
        if use_clahe:
            result = self._apply_clahe(result, clip_limit=clahe_clip)

        return (
            cv2.cvtColor(self._original_for_display, cv2.COLOR_BGR2RGB), 
            cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        )

# --- HELPER: Image to Bytes ---
def convert_image_to_bytes(img_array):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".png", img_bgr)
    if is_success:
        return io.BytesIO(buffer)
    return None

# --- STREAMLIT APP LOGIC ---

st.set_page_config(page_title="Retro Video Enhancer", layout="wide")

# 1. LOAD SETTINGS
init_lang = get_query_param("p_lang", "English")
init_win = get_query_param("p_win", 3, to_int)
init_merge = get_query_param("p_merge", "mean")
init_sharp = get_query_param("p_sharp", 0.5, to_float)
init_clahe = get_query_param("p_clahe", True, to_bool)
init_clip = get_query_param("p_clip", 2.0, to_float)
init_sigma = get_query_param("p_sigma", 2.0, to_float)
init_enhance_live = get_query_param("p_live", False, to_bool)

# Sidebar: Language
lang_code = st.sidebar.selectbox(
    "Language / Sprache", 
    ["English", "Deutsch"], 
    index=0 if init_lang == "English" else 1,
    key="p_lang",
    on_change=lambda: update_query_param("p_lang")
)
txt = TRANSLATIONS[lang_code]

st.title(txt["title"])

# 2. File Upload
uploaded_file = st.file_uploader(txt["upload_label"], type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Prepare Filename base for downloads (Remove spaces, keep it safe)
    base_filename = os.path.splitext(uploaded_file.name)[0].replace(" ", "_")
    
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    # Initialize Engine
    if 'enhancer' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
        with st.spinner(txt["init_spinner"]):
            try:
                st.session_state.enhancer = AdvancedRetroEnhancer(tfile.name)
                st.session_state.current_file = uploaded_file.name
                if 'frame_idx_state' not in st.session_state:
                    st.session_state.frame_idx_state = 0
            except Exception as e:
                st.error(f"{txt['err_load']}: {e}")
                st.stop()

    enhancer = st.session_state.enhancer
    fps = enhancer.fps
    
    # --- NAVIGATION LOGIC ---
    def update_from_slider():
        st.session_state.seconds_state = st.session_state.frame_idx_state / fps
    def update_from_seconds():
        st.session_state.frame_idx_state = int(st.session_state.seconds_state * fps)
    def prev_frame():
        st.session_state.frame_idx_state = max(0, st.session_state.frame_idx_state - 1)
        update_from_slider()
    def next_frame():
        st.session_state.frame_idx_state = min(enhancer.total_frames - 1, st.session_state.frame_idx_state + 1)
        update_from_slider()

    if 'seconds_state' not in st.session_state:
        st.session_state.seconds_state = 0.0

    # 3. Timeline Controls
    st.markdown("---")
    
    time_col1, time_col2 = st.columns([4, 1])
    with time_col1:
        frame_idx = st.slider(txt["frame_idx"], 0, enhancer.total_frames - 1, key="frame_idx_state", on_change=update_from_slider)
    with time_col2:
        seconds = st.number_input(txt["timestamp"], min_value=0.0, max_value=enhancer.total_frames / fps, step=0.1, key="seconds_state", on_change=update_from_seconds)

    _, prev_col, next_col, _ = st.columns([3, 1, 1, 3])
    with prev_col:
        st.button(txt["prev"], on_click=prev_frame, use_container_width=True)
    with next_col:
        st.button(txt["next"], on_click=next_frame, use_container_width=True)

    # 4. Sidebar: Settings
    st.sidebar.header(txt["sidebar_settings"])
    
    window_size = st.sidebar.slider(txt["win_size"], 1, 9, value=init_win, step=2, key="p_win", on_change=lambda: update_query_param("p_win"))

    # --- HORIZONTAL WINDOW VISUALIZATION ---
    if window_size > 1:
        st.sidebar.caption(txt["vis_label"])
        thumbs, caps = enhancer.get_stack_thumbnails(frame_idx, window_size)
        cols = st.sidebar.columns(len(thumbs))
        for i, col in enumerate(cols):
            with col:
                st.image(thumbs[i], caption=caps[i], use_container_width=True)
        st.sidebar.markdown("---")
    # ------------------------------
    
    # Modes
    mode_options = ['mean', 'median', 'gaussian', 'sigma_clip', 'min', 'max']
    try:
        default_mode_idx = mode_options.index(init_merge)
    except ValueError:
        default_mode_idx = 0

    merge_mode = st.sidebar.selectbox(
        txt["merge_mode"], 
        mode_options,
        index=default_mode_idx,
        format_func=lambda x: x.replace('_', ' ').capitalize(),
        key="p_merge", 
        on_change=lambda: update_query_param("p_merge")
    )
    st.sidebar.caption(txt["mode_desc"].get(merge_mode, ""))
    
    sigma_factor = 2.0
    if merge_mode == 'sigma_clip':
        sigma_factor = st.sidebar.slider(
            txt["sigma_factor"], 0.1, 5.0, value=init_sigma, step=0.1,
            key="p_sigma", on_change=lambda: update_query_param("p_sigma")
        )

    st.sidebar.markdown("---")
    
    sharpen_amount = st.sidebar.slider(txt["sharpen"], 0.0, 3.0, value=init_sharp, key="p_sharp", on_change=lambda: update_query_param("p_sharp"))
    use_clahe = st.sidebar.checkbox(txt["use_clahe"], value=init_clahe, key="p_clahe", on_change=lambda: update_query_param("p_clahe"))
    clahe_clip = st.sidebar.slider(txt["clahe_clip"], 1.0, 10.0, value=init_clip, key="p_clip", on_change=lambda: update_query_param("p_clip"))

    st.markdown("---")
    
    # 5. MAIN ACTION: Checkbox
    enhance_live = st.checkbox(
        txt["chk_enhance"], 
        value=init_enhance_live,
        help=txt["enhance_tip"],
        key="p_live",
        on_change=lambda: update_query_param("p_live")
    )
    
    st.markdown("---")

    # 6. Display Images
    raw_preview = enhancer.get_raw_frame(frame_idx)
    
    if raw_preview is not None:
        col1, col2 = st.columns(2)
        
        # --- LEFT: ORIGINAL ---
        with col1:
            st.subheader(txt["header_orig"])
            st.image(raw_preview, use_container_width=True)
            
            btn_orig = convert_image_to_bytes(raw_preview)
            if btn_orig:
                st.download_button(
                    label=txt["dl_orig"],
                    data=btn_orig,
                    file_name=f"{base_filename}_frame{frame_idx:05d}_original.png",
                    mime="image/png"
                )

        # --- RIGHT: ENHANCED ---
        with col2:
            st.subheader(txt["header_enh"])
            
            if enhance_live:
                with st.spinner(txt["proc_spinner"]):
                    _, result = enhancer.process_pipeline(
                        frame_idx=frame_idx,
                        window_size=window_size,
                        merge_mode=merge_mode,
                        sharpen_amount=sharpen_amount,
                        use_clahe=use_clahe,
                        clahe_clip=clahe_clip,
                        sigma_factor=sigma_factor
                    )
                
                if result is not None:
                    st.image(result, use_container_width=True)
                    
                    btn_enh = convert_image_to_bytes(result)
                    if btn_enh:
                        st.download_button(
                            label=txt["dl_enh"],
                            data=btn_enh,
                            file_name=f"{base_filename}_frame{frame_idx:05d}_enhanced.png",
                            mime="image/png"
                        )
                else:
                    st.error(txt["err_process"])
            else:
                st.info(txt["enhance_tip"])
    else:
        st.error(txt["err_process"])
