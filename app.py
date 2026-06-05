# -*- coding: utf-8 -*-
import json
import pickle
import warnings
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Huidlaesie AI Onderzoek",
    page_icon=":microscope:",
    layout="wide",
)


st.markdown("""
<style>
@media screen and (max-width: 768px) {
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 0 !important;
    }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.1rem !important; }
    .stMainBlockContainer {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* Remove fixed-width container constraints on mobile */
    [data-testid="stVerticalBlockBorderWrapper"] > div > div {
        max-width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

DATA = Path("data")
FIGS = DATA / "figures"

TEXT_WIDTH   = 720
PLOT_WIDTH   = 520
PLOT_HEIGHT  = 400
SMALL_HEIGHT = 280

CLASS_NAMES  = ["AKIEC", "BCC", "BKL", "DF", "NV", "MEL", "VASC"]
CLASS_LABELS = {
    "AKIEC": "Actinische Keratose",
    "BCC":   "Basaalcelcarcinoom",
    "BKL":   "Benigne Keratose",
    "DF":    "Dermatofibroom",
    "NV":    "Melanocytaire Naevi (Moedervlek)",
    "MEL":   "Melanoom",
    "VASC":  "Vasculaire Laesie",
}
CLASS_COLORS = {
    "AKIEC": "#8b5cf6",
    "BCC":   "#f59e0b",
    "BKL":   "#2dd4bf",
    "DF":    "#ec4899",
    "NV":    "#10b981",
    "MEL":   "#f43f5e",
    "VASC":  "#1c83e1",
}
MODEL_COLORS = {"B0+MLP": "#1c83e1", "B0+CBM": "#10b981", "B0+ProtoPNet": "#f59e0b"}


@st.cache_data(show_spinner=False)
def load_exp12():
    path = FIGS / "exp12_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def load_bootstrap():
    path = FIGS / "bootstrap_ci.csv"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path)


@st.cache_data(show_spinner=False)
def load_threshold_tuning():
    path = FIGS / "threshold_tuning.csv"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path)


@st.cache_data(show_spinner=False)
def load_clinical_summary():
    path = DATA / "clinical_summary.csv"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path)


@st.cache_data(show_spinner=False)
def load_clinical_safety():
    path = DATA / "clinical_safety_metrics.csv"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path)


@st.cache_data(show_spinner=False)
def load_review_cases():
    path = DATA / "dermatologist_review_cases.csv"
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path)


@st.cache_data(show_spinner=False)
def load_config():
    path = DATA / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_resource(show_spinner="Classificeerder laden...")
def load_classifier():
    clf_path = DATA / "classifier.pkl"
    sc_path  = DATA / "morphology_scaler.pkl"
    cm_path  = DATA / "col_means.pkl"
    if not clf_path.exists():
        return None, None, None
    with open(clf_path, "rb") as f:
        clf_bundle = pickle.load(f)
    clf = clf_bundle["model"] if isinstance(clf_bundle, dict) else clf_bundle
    scaler   = pickle.load(open(sc_path, "rb")) if sc_path.exists() else None
    col_means = pickle.load(open(cm_path, "rb")) if cm_path.exists() else None
    return clf, scaler, col_means


def fig_img(name: str) -> Image.Image | None:
    p = FIGS / name
    return Image.open(p) if p.exists() else None


def data_img(name: str) -> Image.Image | None:
    p = DATA / name
    return Image.open(p) if p.exists() else None


@st.cache_resource(show_spinner="EfficientNet-B0 backbone laden...")
def load_backbone():
    try:
        import timm, torch
        from torchvision import transforms
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        pth_path = DATA / "figures" / "backbone_finetuned.pth"
        if pth_path.exists():
            state = torch.load(str(pth_path), map_location="cpu")
            model.load_state_dict(state, strict=False)
        model.eval()
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return model, tfm
    except Exception:
        return None, None


def extract_b0_embedding(img: Image.Image) -> np.ndarray | None:
    model, tfm = load_backbone()
    if model is None or tfm is None:
        return None
    try:
        import torch
        tensor = tfm(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            emb = model(tensor).squeeze().numpy()
        return emb.astype(np.float32)
    except Exception:
        return None


def extract_morphology_from_pil(img: Image.Image) -> np.ndarray | None:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("inference", DATA / "inference.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.convert("RGB").save(tmp.name)
            feats = mod.extract_morphology(tmp.name)
        Path(tmp.name).unlink(missing_ok=True)
        return feats
    except Exception:
        return None


def run_inference(img: Image.Image):
    clf, scaler, col_means = load_classifier()
    cfg = load_config()
    classes = cfg.get("class_names", CLASS_NAMES)

    morph = extract_morphology_from_pil(img)
    if morph is None or scaler is None:
        morph = np.zeros(44, dtype=np.float32)
    else:
        morph = scaler.transform(morph.reshape(1, -1)).flatten()

    b0 = extract_b0_embedding(img)
    if b0 is None:
        b0 = np.zeros(1280, dtype=np.float32)

    hybrid = np.concatenate([b0, morph]).reshape(1, -1)

    if clf is None:
        raw = np.random.dirichlet(np.ones(len(classes)))
        return {
            "probs":      dict(zip(classes, raw.tolist())),
            "predicted":  classes[int(np.argmax(raw))],
            "morph_feats": morph,
            "b0_emb":     b0,
            "demo_mode":  True,
        }

    try:
        probs_arr = clf.predict_proba(hybrid)[0]
    except Exception:
        probs_arr = np.ones(len(classes)) / len(classes)

    probs = dict(zip(classes, probs_arr.tolist()))
    return {
        "probs":      probs,
        "predicted":  max(probs, key=probs.get),
        "morph_feats": morph,
        "b0_emb":     b0,
        "demo_mode":  False,
    }


def model_comparison_chart(results: dict) -> alt.Chart:
    rows = []
    for name, info in results.get("models", {}).items():
        d = info.get("default", {})
        rows.append({
            "Model":          name,
            "Macro F1":       round(d.get("f1m", 0), 4),
            "AUC":            round(d.get("auc", 0), 4),
            "MEL Recall":     round(d.get("mel_rec", 0), 4),
            "Nauwkeurigheid": round(d.get("acc", 0), 4),
        })
    df = pl.DataFrame(rows)
    domain = list(MODEL_COLORS.keys())
    colors = list(MODEL_COLORS.values())
    return (
        alt.Chart(df, height=SMALL_HEIGHT)
        .mark_point(filled=True, size=200)
        .encode(
            alt.X("AUC:Q", scale=alt.Scale(zero=False)),
            alt.Y("Macro F1:Q", scale=alt.Scale(zero=False)),
            alt.Color("Model:N").scale(domain=domain, range=colors),
            alt.Size("MEL Recall:Q").scale(range=[100, 400]).legend(None),
            tooltip=["Model", "Macro F1", "AUC", "MEL Recall", "Nauwkeurigheid"],
        )
    )


def bootstrap_ci_chart(df: pl.DataFrame) -> alt.Chart:
    if df.is_empty():
        return alt.Chart(pl.DataFrame()).mark_point()
    domain = list(MODEL_COLORS.keys())
    colors = list(MODEL_COLORS.values())
    base = alt.Chart(df, height=SMALL_HEIGHT).encode(
        alt.Y("model:N", title=None),
        alt.Color("model:N").scale(domain=domain, range=colors).legend(None),
    )
    bars = base.mark_errorbar(ticks=True, thickness=2).encode(
        alt.X("f1m_lo:Q", title="Macro F1 (bootstrapped 95% BI)"),
        alt.X2("f1m_hi:Q"),
    )
    points = base.mark_point(filled=True, size=150).encode(
        alt.X("f1m_mean:Q"),
        tooltip=["model",
                 alt.Tooltip("f1m_mean:Q", format=".3f", title="Gem. F1"),
                 alt.Tooltip("f1m_lo:Q",   format=".3f", title="BI laag"),
                 alt.Tooltip("f1m_hi:Q",   format=".3f", title="BI hoog")],
    )
    return bars + points


def per_class_recall_chart(results: dict, model_name: str, tuned: bool) -> alt.Chart:
    info = results.get("models", {}).get(model_name, {})
    key  = "mel_tuned" if tuned else "default"
    recs = info.get(key, {}).get("per_rec", [0] * 7)
    rows = [{"Klasse": cls, "Recall": round(r, 3), "Label": CLASS_LABELS.get(cls, cls)}
            for cls, r in zip(CLASS_NAMES, recs)]
    df = pl.DataFrame(rows).sort("Recall", descending=True)
    domain = CLASS_NAMES
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    return (
        alt.Chart(df, height=PLOT_HEIGHT)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])),
            alt.Y("Klasse:N", sort="-x", title=None),
            alt.Color("Klasse:N").scale(domain=domain, range=colors).legend(None),
            tooltip=["Label", alt.Tooltip("Recall:Q", format=".3f")],
        )
    )


def threshold_tradeoff_chart(thr_df: pl.DataFrame) -> alt.Chart:
    if thr_df.is_empty():
        return alt.Chart(pl.DataFrame()).mark_point()
    domain = list(MODEL_COLORS.keys())
    colors = list(MODEL_COLORS.values())
    return (
        alt.Chart(thr_df, height=PLOT_HEIGHT)
        .mark_point(filled=True, size=120)
        .encode(
            alt.X("mel_rec:Q",  title="MEL Recall (Gevoeligheid)",
                  scale=alt.Scale(domain=[0.3, 1.0])),
            alt.Y("mel_prec:Q", title="MEL Precisie (PPV)",
                  scale=alt.Scale(domain=[0.2, 1.0])),
            alt.Color("model:N").scale(domain=domain, range=colors),
            alt.Shape("status:N").legend(None),
            tooltip=["model",
                     alt.Tooltip("threshold:Q", format=".3f", title="Drempel"),
                     alt.Tooltip("mel_rec:Q",   format=".3f", title="MEL Recall"),
                     alt.Tooltip("mel_prec:Q",  format=".3f", title="MEL Precisie"),
                     alt.Tooltip("f1m:Q",       format=".3f", title="Macro F1")],
        )
    )


def morph_feature_chart(morph_feats: np.ndarray) -> alt.Chart:
    feat_names = [
        "asym_combined", "asym_vertical", "asym_horizontal", "border_irreg",
        "major_axis", "minor_axis", "eccentricity", "area_ratio",
        "convexity_ratio", "waviness", "aspect_ratio", "mask_quality",
        "hue_mean", "hue_std", "hue_entropy", "hue_skewness",
        "sat_mean", "sat_std", "value_mean", "pigment",
        "red_frac", "yellow_frac", "blue_frac", "blue_white_frac",
        "glcm_contrast", "glcm_dissim", "glcm_homogen", "glcm_energy", "glcm_corr",
        "lbp_entropy", "lbp_uniform", "lbp_nonunif",
        "texture_var", "texture_ent", "vessel_density",
        "defect_count", "max_defect_d", "mean_defect_d",
        "curvature_std", "curvature_max", "fractal_dim",
        "border_gran", "perim_norm", "dark_area_ratio",
    ]
    group_map = {}
    for i, n in enumerate(feat_names):
        if i < 12:   group_map[n] = "Vorm"
        elif i < 24: group_map[n] = "Kleur"
        elif i < 35: group_map[n] = "Textuur"
        else:        group_map[n] = "Rand"

    rows = [{"Kenmerk": n, "Waarde": float(v), "Groep": group_map[n]}
            for n, v in zip(feat_names, morph_feats)]
    df = pl.DataFrame(rows).sort("Waarde", descending=True)

    group_colors = {"Vorm": "#8b5cf6", "Kleur": "#f59e0b",
                    "Textuur": "#2dd4bf", "Rand": "#f43f5e"}
    domain = list(group_colors.keys())
    colors = list(group_colors.values())

    return (
        alt.Chart(df, height=max(300, len(feat_names) * 16))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            alt.X("Waarde:Q", title="Genormaliseerde waarde (z-score t.o.v. moedervlek)"),
            alt.Y("Kenmerk:N", sort="-x", title=None),
            alt.Color("Groep:N").scale(domain=domain, range=colors),
            tooltip=["Kenmerk", "Groep", alt.Tooltip("Waarde:Q", format=".3f")],
        )
    )


def probability_bar_chart(probs: dict) -> alt.Chart:
    rows = [{"Klasse": c, "Kans": v, "Label": CLASS_LABELS.get(c, c)}
            for c, v in probs.items()]
    df = pl.DataFrame(rows).sort("Kans", descending=True)
    domain = CLASS_NAMES
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    return (
        alt.Chart(df, height=220)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            alt.X("Kans:Q", scale=alt.Scale(domain=[0, 1]),
                  axis=alt.Axis(format=".0%")),
            alt.Y("Klasse:N", sort="-x", title=None),
            alt.Color("Klasse:N").scale(domain=domain, range=colors).legend(None),
            tooltip=["Label", alt.Tooltip("Kans:Q", format=".1%")],
        )
    )

def per_class_recall_chart_real(safety_df: pl.DataFrame) -> alt.Chart:
    if safety_df.is_empty():
        return alt.Chart(pl.DataFrame()).mark_bar()
    rec_rows = safety_df.filter(pl.col("category") == "Per-Class Recall")
    rows = [
        {
            "Klasse": row["metric"],
            "Recall": round(float(row["value"]), 4),
            "Label": CLASS_LABELS.get(row["metric"], row["metric"]),
        }
        for row in rec_rows.iter_rows(named=True)
    ]
    df = pl.DataFrame(rows).sort("Recall", descending=True)
    domain = CLASS_NAMES
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    return (
        alt.Chart(df, height=PLOT_HEIGHT)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])),
            alt.Y("Klasse:N", sort="-x", title=None),
            alt.Color("Klasse:N").scale(domain=domain, range=colors).legend(None),
            tooltip=["Label", alt.Tooltip("Recall:Q", format=".3f")],
        )
    )


def per_class_f1_chart(safety_df: pl.DataFrame) -> alt.Chart:
    if safety_df.is_empty():
        return alt.Chart(pl.DataFrame()).mark_bar()
    f1_rows = safety_df.filter(pl.col("category") == "Per-Class F1")
    rows = [
        {
            "Klasse": row["metric"],
            "F1-score": round(float(row["value"]), 4),
            "Label": CLASS_LABELS.get(row["metric"], row["metric"]),
        }
        for row in f1_rows.iter_rows(named=True)
    ]
    df = pl.DataFrame(rows).sort("F1-score", descending=True)
    domain = CLASS_NAMES
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    return (
        alt.Chart(df, height=PLOT_HEIGHT)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            alt.X("F1-score:Q", scale=alt.Scale(domain=[0, 1])),
            alt.Y("Klasse:N", sort="-x", title=None),
            alt.Color("Klasse:N").scale(domain=domain, range=colors).legend(None),
            tooltip=["Label", alt.Tooltip("F1-score:Q", format=".3f")],
        )
    )


def operating_points_chart(summary_df: pl.DataFrame) -> alt.Chart:
    if summary_df.is_empty():
        return alt.Chart(pl.DataFrame()).mark_point()
    label_map = {
        "Best F1m: Nonlinear Stack (Exp51:LR)": "Beste F1",
        "Screening thr=0.01 (MelRec~0.93)":     "Screening",
        "Balanced thr=0.07 (Pareto-optimal)":    "Gebalanceerd",
    }
    color_map = {"Beste F1": "#1c83e1", "Screening": "#f43f5e", "Gebalanceerd": "#10b981"}
    rows = []
    for row in summary_df.iter_rows(named=True):
        label = label_map.get(row["operating_point"], row["operating_point"])
        rows.append({
            "Werkpunt": label,
            "MEL Recall":  round(float(row["sensitivity"]), 4),
            "Macro F1":    round(float(row["f1_macro"]), 4),
            "AUC":         round(float(row["auc"]), 4),
            "Precisie":    round(float(row["ppv"]), 4),
            "Drempel":     str(row["threshold"]),
        })
    df = pl.DataFrame(rows)
    domain = ["Beste F1", "Screening", "Gebalanceerd"]
    colors = [color_map[d] for d in domain]
    points = (
        alt.Chart(df, height=SMALL_HEIGHT)
        .mark_point(filled=True, size=250)
        .encode(
            alt.X("MEL Recall:Q", scale=alt.Scale(domain=[0, 1]),
                  title="MEL Recall (Gevoeligheid)"),
            alt.Y("Macro F1:Q", scale=alt.Scale(domain=[0, 1])),
            alt.Color("Werkpunt:N").scale(domain=domain, range=colors),
            tooltip=["Werkpunt", "Drempel",
                     alt.Tooltip("MEL Recall:Q", format=".1%"),
                     alt.Tooltip("Macro F1:Q", format=".3f"),
                     alt.Tooltip("AUC:Q", format=".4f"),
                     alt.Tooltip("Precisie:Q", format=".1%")],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(dy=-14, fontSize=11)
        .encode(
            alt.X("MEL Recall:Q"),
            alt.Y("Macro F1:Q"),
            alt.Text("Werkpunt:N"),
            alt.Color("Werkpunt:N").scale(domain=domain, range=colors).legend(None),
        )
    )
    return points + labels


def clinical_metrics_chart(summary_df: pl.DataFrame) -> alt.Chart:
    """Grouped bar chart: sensitivity, specificity, PPV, NPV per operating point."""
    if summary_df.is_empty():
        return alt.Chart(pl.DataFrame()).mark_bar()
    label_map = {
        "Best F1m: Nonlinear Stack (Exp51:LR)": "Beste F1",
        "Screening thr=0.01 (MelRec~0.93)":     "Screening",
        "Balanced thr=0.07 (Pareto-optimal)":    "Gebalanceerd",
    }
    metric_nl = {
        "sensitivity": "Gevoeligheid",
        "specificity": "Specificiteit",
        "ppv":         "Precisie (PPV)",
        "npv":         "NPV",
    }
    rows = []
    for row in summary_df.iter_rows(named=True):
        label = label_map.get(row["operating_point"], row["operating_point"])
        for col, nl in metric_nl.items():
            rows.append({
                "Werkpunt": label,
                "Maatstaf": nl,
                "Waarde":   round(float(row[col]), 4),
            })
    df = pl.DataFrame(rows)
    wp_domain = ["Beste F1", "Screening", "Gebalanceerd"]
    wp_colors = ["#1c83e1", "#f43f5e", "#10b981"]
    return (
        alt.Chart(df, height=SMALL_HEIGHT)
        .mark_bar()
        .encode(
            alt.X("Maatstaf:N", title=None, axis=alt.Axis(labelAngle=0)),
            alt.Y("Waarde:Q", scale=alt.Scale(domain=[0, 1]), title="Waarde"),
            alt.Color("Werkpunt:N").scale(domain=wp_domain, range=wp_colors),
            alt.XOffset("Werkpunt:N"),
            tooltip=["Werkpunt", "Maatstaf", alt.Tooltip("Waarde:Q", format=".1%")],
        )
    )


def wide_layout():
    with st.container(horizontal_alignment="center"):
        return st.container(
            width=2 * PLOT_WIDTH + 16, horizontal_alignment="center"
        )


def disclaimer():
    st.markdown("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                padding:0.75rem 1rem;margin-top:2rem;color:#64748b;font-size:0.85rem;">
    <strong>&#9432; Disclaimer</strong> &nbsp; Dit is nog niet af.
    </div>
    """, unsafe_allow_html=True)

results12  = load_exp12()
boot_df    = load_bootstrap()
thr_df     = load_threshold_tuning()
review_df  = load_review_cases()
cfg        = load_config()
summary_df = load_clinical_summary()
safety_df  = load_clinical_safety()

model_rows = []
for mname, minfo in results12.get("models", {}).items():
    for mode in ("default", "mel_tuned"):
        d = minfo.get(mode, {})
        if not d:
            continue
        for cls, rec in zip(CLASS_NAMES, d.get("per_rec", [])):
            model_rows.append({
                "Model":          mname,
                "Modus":          "MEL-afgesteld" if mode == "mel_tuned" else "Standaard",
                "Klasse":         cls,
                "Label":          CLASS_LABELS.get(cls, cls),
                "Recall":         round(rec, 4),
                "Macro F1":       round(d.get("f1m", 0), 4),
                "AUC":            round(d.get("auc", 0), 4),
                "MEL Recall":     round(d.get("mel_rec", 0), 4),
                "Nauwkeurigheid": round(d.get("acc", 0), 4),
                "ECE":            round(d.get("ece", 0), 4),
            })
model_df = pl.DataFrame(model_rows) if model_rows else pl.DataFrame()

with wide_layout():
    with st.container(width=TEXT_WIDTH):
        st.title("Huidlaesie AI, _geanalyseerd._")
        st.space()
        """
        Onderzoek voor skin cancer classification met explainable AI op **ISIC 2018** met 7 classes
        en 8k afbeeldingen. Het model heeft een **EfficientNet** basis met vision kenmerken zoals vorm,
        kleur, textuur en rand.

        Drie modelarchitecturen worden vergeleken -- MLP, Concept Bottleneck Model (CBM), en
        ProtoPNet -- waarbij melanoom (:red[MEL]) het primaire klinische doel is.
        """
        st.space()

with wide_layout():
    with st.container(width=TEXT_WIDTH):
        """
        ## Deel I: Modelnauwkeurigheid

        **Hoe vergelijken de onderzochtte pipelinearchitectures?** De scatter hieronder toont de
        macro-F1 van elk model tegenover zijn ROC AUC. De belgrootte geeft melanoomrecall
        weer -- de meest veiligheidskritische statistiek. Foutbalken tonen 95% bootstrapped BI.
        **Gevoeligheid (ook wel _recall_)** — Van alle plekjes die écht melanoom zijn:
            hoeveel herkent de AI? Een gemist melanoom is het gevaarlijkst, dus dit cijfer
            telt hier het zwaarst. Hoger = beter.
 
        **Precisie** — Als de AI "melanoom" zegt: hoe vaak klopt dat dan echt? Lage
            precisie betekent veel vals alarm.
 
        **F1-score** — Eén cijfer dat gevoeligheid en precisie samenvat. Het is alleen
            hoog als de AI weinig gevallen mist én weinig vals alarm geeft. "Macro F1" is
            het gemiddelde over alle 7 typen, waarbij elk type even zwaar telt — ook de
            zeldzame. (0 = slecht, 1 = perfect.)
 
        **ROC AUC** — Hoe goed kan de AI twee groepen uit elkaar houden (bijvoorbeeld
            melanoom vs. geen melanoom), ongeacht hoe streng je de grens legt?
            0,5 betekent puur gokken, 1,0 betekent perfect onderscheid.
        """
        st.space()

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("F1-score per klasse")
        st.altair_chart(per_class_f1_chart(safety_df), use_container_width=True)

    with cols[1]:
        st.subheader("Werkpunten: gevoeligheid vs. F1")
        st.altair_chart(operating_points_chart(summary_df), use_container_width=True)

    st.space()

    with st.container(width=TEXT_WIDTH):
        """
        **Verwarringmatrices en ROC-curves** van de gehouden testset (1.502 afbeeldingen).
        Blauwe diagonaal = correcte voorspellingen. De ROC-curve toont de
        gevoeligheid/specificiteit-afweging voor melanoom bij alle beslissingsdrempels.
        """

    st.space()
    cols2 = st.columns(2, border=True)
    with cols2[0]:
        img_cm = fig_img("confusion_matrices.png")
        if img_cm:
            st.image(img_cm, caption="Verwarringmatrices -- rijen: werkelijke klasse, kolommen: voorspeld", use_container_width=True)
    with cols2[1]:
        img_roc = fig_img("melanoma_roc.png")
        if img_roc:
            st.image(img_roc, caption="Melanoom ROC-curves met AUC en werkpunten", use_container_width=True)

    st.space()

    with st.container(width=TEXT_WIDTH):
        """
        **Kalibratie van kansen** -- Een perfect gekalibreerd model volgt de diagonaal.
        """

    rel_img = fig_img("reliability_diagrams.png")
    if rel_img:
        st.image(rel_img, caption="Betrouwbaarheidsdiagrammen voor en na temperatuurscaling", use_container_width=True)

    disclaimer()

with wide_layout():
    st.space("large")
    with st.container(width=TEXT_WIDTH):
        """
        ## Deel II: Prestaties per klasse

        **Welke laesietypen pakt de AI het best -- en het slechtst?** Recall geeft aan
        hoeveel gevallen van elk type correct worden herkend. Melanoom (:red[MEL]) heeft de
        laagste recall -- gemiste melanomen zijn het gevaarlijkste faaltype.
        """
        st.space()

    cols3 = st.columns(2, border=True)
    with cols3[0]:
        st.subheader("Recall per klasse (testset)")
        st.altair_chart(per_class_recall_chart_real(safety_df), use_container_width=True)

    with cols3[1]:
        st.subheader("Samenvattende statistieken")
        st.space("small")
        if not summary_df.is_empty():
            best     = summary_df.row(0, named=True)
            screen   = summary_df.row(1, named=True)
            balanced = summary_df.row(2, named=True)
            m1, m2 = st.columns(2)
            m1.metric("Macro F1 (beste model)", f"{best['f1_macro']:.3f}")
            m2.metric("ROC AUC",                f"{best['auc']:.4f}")
            m1.metric("MEL Recall (argmax)",    f"{best['sensitivity']:.1%}")
            m2.metric("MEL Recall (screening)", f"{screen['sensitivity']:.1%}")
            m1.metric("MEL Precisie",           f"{best['ppv']:.1%}")
            m2.metric("NPV (screening)",        f"{screen['npv']:.3f}",
                      help="Negatief Voorspellende Waarde: kans dat negatief resultaat klopt")

    st.space()

    with st.container(width=TEXT_WIDTH):
        """
        **Klinische maatstaven per werkpunt** -- gevoeligheid (MEL recall), specificiteit,
        precisie (PPV) en negatief voorspellende waarde (NPV) voor elk van de drie drempels.
        Screening maximaliseert gevoeligheid; Beste F1 balanceert alle maatstaven.
        """
    st.altair_chart(clinical_metrics_chart(summary_df), use_container_width=True)

    disclaimer()

with wide_layout():
    st.space("large")
    with st.container(width=TEXT_WIDTH):
        """
        ## Deel III: Kenmerkenanalyse

        **Wat meet de AI eigenlijk?** De 44 morfologische kenmerken coderen klinische kennis
        uit de ABCD-regel (Asymmetrie, Rand, Kleur, Dermoscopische structuren). Hieronder
        staan de resultaten rechtstreeks uit het Kaggle-experiment.
        """
        st.space()

    # Training curves
    train_img = data_img("__results___10_3.png")
    if train_img:
        st.image(train_img,
                 caption="Trainingscurves: verlies en nauwkeurigheid per epoch voor elk model",
                 use_container_width=True)

    st.space()

    # Morphology extraction + threshold sweep
    cols4 = st.columns(2, border=True)
    with cols4[0]:
        morph_img = data_img("__results___10_4.png")
        if morph_img:
            st.image(morph_img,
                     caption="Morfologische kenmerken: extractiediagram (ABCD-regel)",
                     use_container_width=True)
    with cols4[1]:
        thr_img = data_img("__results___10_5.png")
        if thr_img:
            st.image(thr_img,
                     caption="Drempelzoeking: gevoeligheid en specificiteit als functie van de drempel",
                     use_container_width=True)

    st.space()

    # Calibration curves
    with st.container(width=TEXT_WIDTH):
        """
        **Kalibratie** -- een goed gekalibreerd model heeft kansen die overeenkomen met de
        werkelijke frequentie. Temperatuurscaling (T = 0,58) corrigeert het oververtrouwen
        van het netwerk.
        """
    cal_img = data_img("__results___10_7.png")
    if cal_img:
        st.image(cal_img,
                 caption="Kalibratie voor en na temperatuurscaling",
                 use_container_width=True)

    disclaimer()

with wide_layout():
    st.space("large")
    with st.container(width=TEXT_WIDTH):
        """
        ## Deel IV: Analyseer een afbeelding

        Upload een huidlaesie afbeelding. De pipeline extraheert **44 morfologische kenmerken**
        in real-time (Otsu-segmentatie + GLCM + LBP + randanalyse), combineert ze met
        **EfficientNet-B0 inbeddingen**, en geeft de hybride vector door aan de
        geimplementeerde classificeerder.
        """
        st.space()

    uploaded = st.file_uploader(
        "Upload een huidlaesie afbeelding (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        with st.spinner("Analyse wordt uitgevoerd..."):
            result = run_inference(img)

        probs     = result["probs"]
        predicted = result["predicted"]
        morph     = result["morph_feats"]
        demo      = result.get("demo_mode", False)

        if demo:
            st.info("Demomodus actief (backbone of classificeerder niet beschikbaar). Kansen zijn willekeurig gegenereerd.")

        st.space()

        cols_inf = st.columns(2, border=True)
        with cols_inf[0]:
            st.subheader("Invoerafbeelding")
            st.image(img, use_container_width=True)

        with cols_inf[1]:
            st.subheader("GradCAM aandachtskaart (voorbeeld)")
            gc_img = data_img("gradcam.png")
            if gc_img:
                st.image(gc_img,
                         caption="Typische GradCAM heatmap -- rood = hoogste modelaandacht",
                         use_container_width=True)
            else:
                st.info("GradCAM afbeelding niet gevonden.")

        st.space()

        cols_pred = st.columns([1, 2], border=True)
        with cols_pred[0]:
            conf      = round(probs[predicted] * 100, 1)
            mel_prob  = probs.get("MEL", 0)
            pred_label = CLASS_LABELS.get(predicted, predicted)

            st.subheader("Voorspelling")
            st.metric("Voorspelde klasse", pred_label)
            st.metric("Betrouwbaarheid", f"{conf}%")
            st.metric("Melanoomkans", f"{mel_prob:.1%}",
                      delta="Verhoogd" if mel_prob > 0.3 else "Laag risico",
                      delta_color="inverse" if mel_prob > 0.3 else "normal")

        with cols_pred[1]:
            st.subheader("Kansenverdeling per klasse")
            st.altair_chart(probability_bar_chart(probs), use_container_width=True)

        st.space()

        with st.container(border=True):
            st.subheader("Morfologische kenmerken (44 kenmerken, z-gescoord)")
            st.caption(
                "Elke balk toont hoeveel deze afbeelding afwijkt van de benigne moedervlek basislijn. "
                "Waarden buiten +/-1,5 zijn klinisch opvallend."
            )
            st.altair_chart(morph_feature_chart(morph), use_container_width=True)

        feat_names = [
            "asym_combined", "asym_vertical", "asym_horizontal", "border_irreg",
            "major_axis", "minor_axis", "eccentricity", "area_ratio",
            "convexity_ratio", "waviness", "aspect_ratio", "mask_quality",
            "hue_mean", "hue_std", "hue_entropy", "hue_skewness",
            "sat_mean", "sat_std", "value_mean", "pigment",
            "red_frac", "yellow_frac", "blue_frac", "blue_white_frac",
            "glcm_contrast", "glcm_dissim", "glcm_homogen", "glcm_energy", "glcm_corr",
            "lbp_entropy", "lbp_uniform", "lbp_nonunif",
            "texture_var", "texture_ent", "vessel_density",
            "defect_count", "max_defect_d", "mean_defect_d",
            "curvature_std", "curvature_max", "fractal_dim",
            "border_gran", "perim_norm", "dark_area_ratio",
        ]
        groups = ["Vorm"] * 12 + ["Kleur"] * 12 + ["Textuur"] * 11 + ["Rand"] * 9
        feat_df = pl.DataFrame({
            "Kenmerk":          feat_names,
            "Groep":            groups,
            "Waarde (z-score)": [round(float(v), 4) for v in morph],
        })
        st.dataframe(
            feat_df,
            height=300,
            use_container_width=True,
            column_config={
                "Waarde (z-score)": st.column_config.ProgressColumn(
                    min_value=-4, max_value=4, format="%.3f",
                ),
                "Groep": st.column_config.SelectboxColumn(
                    options=["Vorm", "Kleur", "Textuur", "Rand"],
                ),
            },
            hide_index=True,
        )

    disclaimer()

with wide_layout():
    st.space("large")
    with st.container(width=TEXT_WIDTH):
        """
        ## Deel V: Klinische beoordelingsgevallen

        **80 gevallen samengesteld voor dermatoloogbeoordeling**, verdeeld over vier cohorten:
        gemakkelijke melanomen (hoog vertrouwen), gemiste melanomen (fout-negatief),
        fout-positieven, en gevallen met hoge onzekerheid. Elke rij bevat de top 3
        morfologische kenmerken (als SHAP z-scores) die de voorspelling stuurden.
        """
        st.space()

    if not review_df.is_empty():
        cohorts = ["Alles"] + sorted(review_df["cohort"].unique().to_list())
        sel = st.selectbox(
            "Filter op cohort",
            cohorts,
            format_func=lambda x: x.replace("_", " ") if x != "Alles" else "Alle cohorten (80 gevallen)",
        )
        shown = review_df if sel == "Alles" else review_df.filter(pl.col("cohort") == sel)

        st.dataframe(
            shown,
            use_container_width=True,
            height=400,
            hide_index=True,
            column_config={
                "cohort":                st.column_config.TextColumn("Cohort"),
                "image_id":              st.column_config.TextColumn("Geval-ID", pinned=True),
                "true_diagnosis":        st.column_config.SelectboxColumn(
                                             "Werkelijke Dx", options=CLASS_NAMES),
                "prediction":            st.column_config.SelectboxColumn(
                                             "AI Voorspelling", options=CLASS_NAMES),
                "confidence":            st.column_config.ProgressColumn(
                                             "Betrouwbaarheid", min_value=0, max_value=1,
                                             format="%.1%"),
                "melanoma_probability":  st.column_config.ProgressColumn(
                                             "MEL Kans", min_value=0, max_value=1,
                                             format="%.1%", color="auto"),
                "calibrated_mel_prob":   st.column_config.ProgressColumn(
                                             "Gekalibreerde MEL Kans", min_value=0,
                                             max_value=1, format="%.1%"),
                "entropy":               st.column_config.NumberColumn(
                                             "Onzekerheid", format="%.3f"),
                "top_morphology_features": st.column_config.TextColumn("Top 3 Kenmerken"),
                "nearest_neighbor_ids":  None,
            },
        )

        st.space()
        summary = (
            review_df
            .group_by("cohort")
            .agg([
                pl.len().alias("N"),
                pl.col("melanoma_probability").cast(pl.Float64).mean().round(3).alias("Gem. MEL Kans"),
                pl.col("entropy").cast(pl.Float64).mean().round(3).alias("Gem. Onzekerheid"),
            ])
            .sort("cohort")
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Klinische beoordelingsdata niet gevonden in data/dermatologist_review_cases.csv")

    
