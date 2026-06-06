#!/usr/bin/env python3
"""
Standalone inference script for the ISIC skin lesion classifier.
Produced by Experiment 51 (fine-tuned EfficientNet-B0 + hybrid morphology).

Usage:
    python inference.py path/to/image.jpg
    python inference.py /deploy/dir/ image1.jpg image2.jpg

Python dependencies:
    torch, timm, numpy, scikit-learn, scikit-image, opencv-python, Pillow,
    xgboost or lightgbm (whichever model was selected during training)
"""

import json
import pickle
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# 44 Morphology feature names (must match training order exactly)
# --------------------------------------------------------------------------
FEAT_NAMES = [
    # Shape (12)
    'asym_combined', 'asym_vertical', 'asym_horizontal', 'border_irreg',
    'major_axis', 'minor_axis', 'eccentricity', 'area_ratio',
    'convexity_ratio', 'waviness', 'aspect_ratio', 'mask_quality',
    # Color (12)
    'hue_mean', 'hue_std', 'hue_entropy', 'hue_skewness',
    'sat_mean', 'sat_std', 'value_mean', 'pigment',
    'red_frac', 'yellow_frac', 'blue_frac', 'blue_white_frac',
    # Texture (11)
    'glcm_contrast', 'glcm_dissim', 'glcm_homogen', 'glcm_energy', 'glcm_corr',
    'lbp_entropy', 'lbp_uniform', 'lbp_nonunif',
    'texture_var', 'texture_ent', 'vessel_density',
    # Border (9)
    'defect_count', 'max_defect_d', 'mean_defect_d',
    'curvature_std', 'curvature_max', 'fractal_dim',
    'border_gran', 'perim_norm', 'dark_area_ratio',
]
assert len(FEAT_NAMES) == 44, f'Expected 44 features, got {len(FEAT_NAMES)}'


# --------------------------------------------------------------------------
# Morphology feature extraction (replicates Exp25 _extract_features25)
# --------------------------------------------------------------------------
def extract_morphology(img_path, size=128):
    """
    Extract 44 morphological features from a skin lesion image.

    Parameters
    ----------
    img_path : str or Path
        Path to the input image.
    size : int
        Resize target (default 128).

    Returns
    -------
    np.ndarray of shape (44,), dtype float32.  NaN replaced by 0.
    """
    try:
        from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

        img    = Image.open(img_path)
        img_np = np.array(img.resize((size, size)).convert('RGB'))
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hsv    = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Segmentation: Otsu on inverted V-channel + morphological cleanup
        v_ch = hsv[:, :, 2]
        _, mask = cv2.threshold(255 - v_ch, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_lab > 1:
            best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            mask = (labels == best).astype(np.uint8) * 255
        area  = float(mask.sum() // 255)
        total = float(size * size)
        if area < 100:
            mask = np.ones((size, size), dtype=np.uint8) * 255
            area = total
        mask_bool = mask > 0

        # ---- GROUP 1: SHAPE (12) ----
        h, w = size, size
        top    = mask_bool[:h // 2, :]
        bottom = mask_bool[h // 2:, :][::-1, :]
        left   = mask_bool[:, :w // 2]
        right  = mask_bool[:, w // 2:][:, ::-1]
        asym_v = float(np.logical_xor(top,  bottom[:top.shape[0],   :]).mean())
        asym_h = float(np.logical_xor(left, right[:, :left.shape[1]]).mean())
        asym_combined = (asym_v + asym_h) / 2.0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt   = max(contours, key=cv2.contourArea)
            perim = cv2.arcLength(cnt, True)
            compact = (perim ** 2) / (4 * np.pi * max(area, 1.0))
            border_irreg = float(np.clip((compact - 1.0) / 9.0, 0, 1))
            hull_pts  = cv2.convexHull(cnt, returnPoints=True)
            hull_area = float(cv2.contourArea(hull_pts))
            cnt_area  = float(cv2.contourArea(cnt))
            convexity_ratio = float(cnt_area / max(hull_area, 1.0))
            hull_perim = cv2.arcLength(hull_pts, True)
            waviness   = float(perim / max(hull_perim, 1e-6))
            perim_norm = float(perim / (4.0 * size))
            border_gran = float(perim / (np.sqrt(max(area, 1.0)) * size))
            if len(cnt) >= 5:
                (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
                major_axis   = float(max(ma, mi)) / size
                minor_axis   = float(min(ma, mi)) / size
                eccentricity = float(
                    np.sqrt(max(0.0, 1.0 - (minor_axis / max(major_axis, 1e-6)) ** 2))
                )
                aspect_ratio = float(max(ma, mi) / max(min(ma, mi), 1e-6))
            else:
                major_axis = minor_axis = eccentricity = 0.0
                aspect_ratio = 1.0
        else:
            cnt = None
            border_irreg = major_axis = minor_axis = eccentricity = 0.0
            convexity_ratio = 1.0; waviness = 1.0; aspect_ratio = 1.0
            perim = 0.0; perim_norm = 0.0; border_gran = 0.0
        area_ratio   = float(np.clip(area / total, 0, 1))
        mask_quality = 1.0 if area >= 100 else 0.0

        # ---- GROUP 2: COLOR (12) ----
        h_ch = hsv[:, :, 0][mask_bool].astype(float)
        s_ch = hsv[:, :, 1][mask_bool].astype(float)
        v_px = hsv[:, :, 2][mask_bool].astype(float)
        if len(h_ch) > 0:
            hue_mean = float(h_ch.mean() / 180.0)
            hue_std  = float(h_ch.std()  / 90.0)
            hist_h, _ = np.histogram(h_ch, bins=32, range=(0, 180))
            hist_h    = hist_h / (hist_h.sum() + 1e-9)
            hue_entropy  = -float((hist_h * np.log(hist_h + 1e-9)).sum()) / np.log(32)
            hue_skewness = float(
                np.clip((h_ch.mean() - np.median(h_ch)) / (h_ch.std() + 1e-9), -3, 3)
            ) / 3.0
            sat_mean   = float(s_ch.mean() / 255.0)
            sat_std    = float(s_ch.std()  / 128.0)
            value_mean = float(v_px.mean() / 255.0)
            pigment    = 1.0 - value_mean
            red_frac        = float(((h_ch < 15) | (h_ch > 165)).mean())
            yellow_frac     = float(((h_ch >= 15) & (h_ch < 60)).mean())
            blue_frac       = float(((h_ch >= 90) & (h_ch < 130)).mean())
            blue_white_frac = float(((s_ch < 80) & (v_px > 180)).mean())
        else:
            hue_mean = hue_std = hue_entropy = hue_skewness = 0.0
            sat_mean = sat_std = 0.0
            value_mean = 0.5; pigment = 0.5
            red_frac = yellow_frac = blue_frac = blue_white_frac = 0.0

        # ---- GROUP 3: TEXTURE (11) ----
        gray_q = np.clip(gray // 8, 0, 31).astype(np.uint8)
        try:
            glcm = graycomatrix(
                gray_q, distances=[1],
                angles=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=32, symmetric=True, normed=True,
            )
            glcm_contrast = float(graycoprops(glcm, 'contrast').mean())
            glcm_dissim   = float(graycoprops(glcm, 'dissimilarity').mean())
            glcm_homogen  = float(graycoprops(glcm, 'homogeneity').mean())
            glcm_energy   = float(graycoprops(glcm, 'energy').mean())
            glcm_corr     = float(graycoprops(glcm, 'correlation').mean())
        except Exception:
            glcm_contrast = glcm_dissim = 0.0
            glcm_homogen = glcm_energy = 0.5; glcm_corr = 0.0
        try:
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_m = lbp[mask_bool]
            hist_lbp, _ = np.histogram(lbp_m, bins=10, range=(0, 10))
            hist_lbp = hist_lbp.astype(float) / (hist_lbp.sum() + 1e-9)
            lbp_entropy = -float((hist_lbp * np.log(hist_lbp + 1e-9)).sum()) / np.log(10)
            lbp_uniform = float(hist_lbp[:9].sum())
            lbp_nonunif = float(hist_lbp[9])
        except Exception:
            lbp_entropy = lbp_uniform = lbp_nonunif = 0.0
        if mask_bool.any():
            vals = gray[mask_bool].astype(float)
            texture_var = float(np.clip(vals.var() / (128.0 ** 2), 0, 1))
            ht_hist, _ = np.histogram(vals, bins=32, range=(0, 256))
            ht_hist = ht_hist / (ht_hist.sum() + 1e-9)
            texture_ent = -float((ht_hist * np.log(ht_hist + 1e-9)).sum()) / np.log(32)
        else:
            texture_var = texture_ent = 0.0
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tophat = cv2.morphologyEx(255 - gray, cv2.MORPH_TOPHAT, kernel3)
        vessel_density = float(tophat[mask_bool].mean() / 255.0) if mask_bool.any() else 0.0

        # ---- GROUP 4: BORDER (9) ----
        defect_count = max_defect_d = mean_defect_d = 0.0
        curvature_std = curvature_max = 0.0
        if cnt is not None and len(cnt) > 5:
            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            try:
                defs = cv2.convexityDefects(cnt, hull_idx)
                if defs is not None and len(defs) > 0:
                    depths = [d[0][3] / 256.0 for d in defs]
                    defect_count  = float(min(len(defs), 30)) / 30.0
                    max_defect_d  = float(max(depths)) / size
                    mean_defect_d = float(np.mean(depths)) / size
            except Exception:
                pass
            pts = cnt.squeeze().astype(float)
            if pts.ndim == 2 and len(pts) > 10:
                d1 = np.diff(pts, axis=0)
                d2 = np.diff(d1, axis=0)
                if len(d2) > 0:
                    cross = np.abs(d2[:, 0] * d1[:-1, 1] - d2[:, 1] * d1[:-1, 0])
                    denom = (d1[:-1] ** 2).sum(1) ** 1.5 + 1e-9
                    curv  = cross / denom
                    curvature_std = float(np.clip(curv.std(), 0, 1e6) / 1e6)
                    curvature_max = float(np.clip(np.percentile(curv, 95), 0, 1e6) / 1e6)
        border_px  = cv2.Canny(mask, 100, 200) > 0
        box_sizes  = [2, 4, 8, 16, 32, 64]
        box_counts = []
        for bs in box_sizes:
            bp = border_px
            ht_bp, wd_bp = bp.shape
            ht_t = (ht_bp // bs) * bs
            wd_t = (wd_bp // bs) * bs
            blk  = bp[:ht_t, :wd_t].reshape(ht_t // bs, bs, wd_t // bs, bs)
            box_counts.append(max(int(blk.any(axis=(1, 3)).sum()), 1))
        log_s  = np.log(1.0 / np.array(box_sizes, float))
        log_c  = np.log(np.array(box_counts, float))
        slope, _ = np.polyfit(log_s, log_c, 1)
        fractal_dim = float(np.clip(slope, 1.0, 2.0) - 1.0)
        dark_area_ratio = float((v_px < 50).mean()) if len(v_px) > 0 else 0.0

        feat = np.array([
            asym_combined, asym_v, asym_h, border_irreg,
            major_axis, minor_axis, eccentricity, area_ratio,
            convexity_ratio, waviness, aspect_ratio, mask_quality,
            hue_mean, hue_std, hue_entropy, hue_skewness,
            sat_mean, sat_std, value_mean, pigment,
            red_frac, yellow_frac, blue_frac, blue_white_frac,
            glcm_contrast, glcm_dissim, glcm_homogen, glcm_energy, glcm_corr,
            lbp_entropy, lbp_uniform, lbp_nonunif,
            texture_var, texture_ent, vessel_density,
            defect_count, max_defect_d, mean_defect_d,
            curvature_std, curvature_max, fractal_dim,
            border_gran, perim_norm, dark_area_ratio,
        ], dtype=np.float32)
        feat = np.where(np.isfinite(feat), feat, 0.0)
        return feat
    except Exception as e:
        print(f'[extract_morphology] Error on {img_path}: {e}')
        return np.zeros(44, dtype=np.float32)


# --------------------------------------------------------------------------
# DermPredictor class
# --------------------------------------------------------------------------
class DermPredictor:
    """
    End-to-end skin lesion classifier.

    Parameters
    ----------
    deploy_dir : str or Path
        Directory containing: backbone_finetuned.pth, classifier.pkl,
        morphology_scaler.pkl, col_means.pkl, config.json
    """

    def __init__(self, deploy_dir):
        self.deploy_dir = Path(deploy_dir)
        self._load_artefacts()
        self._build_transform()
        print(f'[DermPredictor] Ready.  Model={self.model_name}  Classes={self.class_names}')

    def _load_artefacts(self):
        import torch
        import timm
        with open(self.deploy_dir / 'config.json') as f:
            cfg = json.load(f)
        self.class_names    = cfg['class_names']
        self.mel_idx        = cfg['mel_idx']
        self.n_classes      = cfg['n_classes']
        self.feat_names     = cfg['feat_names']
        self.T              = cfg['T']
        self.thr_default    = cfg['threshold_default']
        self.thr_pareto     = cfg['threshold_pareto']
        self.thr_screening  = cfg['threshold_screening']
        self.model_name     = cfg['model_name']
        self.image_size     = cfg['image_size']
        self.normalize_mean = cfg['normalize_mean']
        self.normalize_std  = cfg['normalize_std']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        backbone_path = self.deploy_dir / 'backbone_finetuned.pth'
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        state = torch.load(backbone_path, map_location=self.device)
        state = {k: v for k, v in state.items()
                 if not k.startswith('classifier') and not k.startswith('head')}
        self.backbone.load_state_dict(state, strict=False)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        with open(self.deploy_dir / 'classifier.pkl', 'rb') as f:
            clf_pkg = pickle.load(f)
        self.classifier = clf_pkg['model'] if isinstance(clf_pkg, dict) else clf_pkg
        with open(self.deploy_dir / 'morphology_scaler.pkl', 'rb') as f:
            self.morph_scaler = pickle.load(f)
        with open(self.deploy_dir / 'col_means.pkl', 'rb') as f:
            self.col_means = pickle.load(f)

    def _build_transform(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.normalize_mean, self.normalize_std),
        ])

    def extract_embedding(self, img_path):
        """
        Extract 1280-dim EfficientNet-B0 embedding for a single image.

        Parameters
        ----------
        img_path : str or Path

        Returns
        -------
        np.ndarray of shape (1280,), dtype float32
        """
        import torch
        img = Image.open(img_path).convert('RGB')
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.backbone(tensor)
        return emb.squeeze(0).cpu().float().numpy()

    def _softmax(self, logits):
        logits = logits - logits.max()
        e = np.exp(logits)
        return e / e.sum()

    def predict(self, img_path):
        """
        Predict skin lesion class for a single image.

        Parameters
        ----------
        img_path : str or Path

        Returns
        -------
        dict with keys:
            image_id             : str (filename)
            prediction           : str (class name)
            confidence           : float (probability of predicted class)
            top_3_classes        : list[str]
            top_3_probs          : list[float]
            melanoma_probability : float (calibrated)
            review_flag          : bool (True if p(MEL) >= screening threshold)
            all_probs            : dict {class_name: probability}
        """
        img_path = Path(img_path)
        emb = self.extract_embedding(img_path)
        raw_morph = extract_morphology(img_path)
        raw_morph = raw_morph.copy().astype(np.float64)
        for j in range(len(raw_morph)):
            if not np.isfinite(raw_morph[j]):
                raw_morph[j] = self.col_means[j]
        morph_sc = self.morph_scaler.transform(raw_morph.reshape(1, -1))
        x_hyb = np.concatenate([emb.reshape(1, -1), morph_sc], axis=1)
        raw_prob = self.classifier.predict_proba(x_hyb)[0]
        if hasattr(self.classifier, 'decision_function'):
            logits = self.classifier.decision_function(x_hyb)[0]
        else:
            eps = 1e-9
            logits = np.log(np.clip(raw_prob, eps, 1.0))
        calib_prob  = self._softmax(logits / max(self.T, 1e-3))
        pred_idx    = int(calib_prob.argmax())
        pred_class  = self.class_names[pred_idx]
        confidence  = float(calib_prob[pred_idx])
        mel_prob    = float(calib_prob[self.mel_idx])
        review_flag = mel_prob >= self.thr_screening
        top3_idx   = np.argsort(calib_prob)[::-1][:3]
        top3_names = [self.class_names[i] for i in top3_idx]
        top3_probs = [float(calib_prob[i]) for i in top3_idx]
        all_probs  = {self.class_names[i]: float(calib_prob[i])
                      for i in range(self.n_classes)}
        return {
            'image_id':            img_path.name,
            'prediction':          pred_class,
            'confidence':          confidence,
            'top_3_classes':       top3_names,
            'top_3_probs':         top3_probs,
            'melanoma_probability': mel_prob,
            'review_flag':         review_flag,
            'all_probs':           all_probs,
        }


# --------------------------------------------------------------------------
# CLI example
# --------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python inference.py <image_path> [image_path ...]')
        print('       python inference.py /path/to/deploy/ image1.jpg image2.jpg')
        sys.exit(1)
    deploy_dir = Path(sys.argv[1])
    if deploy_dir.is_dir() and (deploy_dir / 'config.json').exists():
        img_paths = sys.argv[2:]
    else:
        deploy_dir = Path(__file__).parent
        img_paths  = sys.argv[1:]
    if not img_paths:
        print('No image paths provided.')
        sys.exit(1)
    predictor = DermPredictor(deploy_dir)
    print()
    print(f'  {"Image":<40}  {"Pred":<8}  {"Conf":>6}  {"MEL_p":>6}  {"Review":>7}')
    print('  ' + '-' * 74)
    for img_path in img_paths:
        try:
            r = predictor.predict(img_path)
            flag_str = 'REVIEW' if r['review_flag'] else 'ok'
            top3_str = ', '.join(f'{c}={p:.3f}'
                                 for c, p in zip(r['top_3_classes'], r['top_3_probs']))
            print(f'  {r["image_id"]:<40}  {r["prediction"]:<8}  '
                  f'{r["confidence"]:>6.4f}  {r["melanoma_probability"]:>6.4f}  '
                  f'{flag_str:>7}')
            print(f'    Top-3: {top3_str}')
            print()
        except Exception as exc:
            print(f'  ERROR on {img_path}: {exc}')
