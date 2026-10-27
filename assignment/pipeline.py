"""Conservative-care targeting pipeline for Meridian MSK patients.

    uv run python -m assignment.pipeline train
    uv run python -m assignment.pipeline score

Prevalent multi-snapshot design: patients are indexed at quarterly snapshot dates
while in the funnel, so funnel features vary and the risk model can discriminate.
Train and score use the same snapshot form. A patient may recur across snapshots, so
grouped cross-validation and a patient-level dev/test split prevent leakage.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from assignment.utils import validate_targets

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
OUT_DIR = Path("out")

DATA_START = pd.Timestamp("2023-07-01")
DATA_END = pd.Timestamp("2026-06-30")
SCORE_T0 = pd.Timestamp("2026-07-01")
H = pd.Timedelta(days=365)
Y1 = pd.Timedelta(days=365)
Q3 = pd.Timedelta(days=90)

# quarterly training snapshots: in the funnel, divertible, 12mo outcome observable
TRAIN_SNAPSHOTS = list(pd.date_range("2024-01-01", "2025-04-01", freq="QS"))

# Risk-model features (8), pruned from ~40 on both AUC and calibration. obs_window is
# kept despite ~0 AUC because it protects the probabilities (dropping it worsens ECE).
# Other columns are computed in build_features but used only by the score layer.
CLINICAL_FEATURES = [
    "age", "obs_window",                                   # demographics / data-completeness
    "bmi", "comorbidity_count",                             # health burden
    "rx_rate",                                              # pharmacy / treatment intensity
    "max_stage",                                            # ordinal funnel depth (LBP..consult)
    "days_since_first_lbp", "escalation_events_90d",        # chronicity + recent acceleration
]


class CalibratedModel:
    """Base estimator + isotonic map fit on out-of-fold predictions (group-safe)."""

    def __init__(self, base, iso):
        self.base = base
        self.iso = iso

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        return self.iso.transform(raw)


# ----------------------------------------------------------------------------- data
def load_data() -> dict:
    pat = pd.read_csv(DATA_DIR / "meridian_health_patients.csv",
                      parse_dates=["coverage_start", "coverage_end"])
    enc = pd.read_csv(DATA_DIR / "meridian_health_encounters.csv",
                      parse_dates=["event_date"])
    enr = pd.read_csv(DATA_DIR / "meridian_health_enrollments.csv",
                      parse_dates=["enrollment_date"])
    out = pd.read_csv(DATA_DIR / "meridian_health_outreach.csv",
                      parse_dates=["timestamp"])
    # the data dictionary ships claims as .csv.gz; accept either form (pandas reads gzip transparently)
    claims_path = DATA_DIR / "data_web_claims.csv"
    if not claims_path.exists():
        claims_path = DATA_DIR / "data_web_claims.csv.gz"
    clm = pd.read_csv(
        claims_path,
        usecols=["patient_token", "service_date", "place_of_service",
                 "provider_specialty", "code_system", "code", "claim_status"],
        parse_dates=["service_date"], dtype={"code": "string"},
    )
    pat["coverage_end"] = pat["coverage_end"].fillna(pd.Timestamp("2200-01-01"))
    return {"pat": pat, "enc": enc, "enr": enr, "out": out, "clm": clm}


# --------------------------------------------------------------------------- cohort
def build_cohort(data: dict, mode: str) -> pd.DataFrame:
    """Multi-snapshot (train) or single 2026-07 snapshot (score). One row per
    (patient, snapshot) where the patient is in the funnel and still divertible."""
    pat, enc = data["pat"], data["enc"]
    first_lbp = (enc[enc.event_category == "LBP Encounter"]
                 .groupby("patient_token").event_date.min().rename("first_lbp"))
    surg = enc[(enc.event_category == "Spine Surgery") &
               (enc.event_status == "Performed")][["patient_token", "event_date"]]
    first_surg = surg.groupby("patient_token").event_date.min().rename("first_surg")
    base = pat.merge(first_lbp, on="patient_token").merge(first_surg, on="patient_token", how="left")
    surg_by = surg.groupby("patient_token").event_date.apply(list).to_dict()

    dates = [SCORE_T0] if mode == "score" else TRAIN_SNAPSHOTS
    rows = []
    for T0 in dates:
        df = base[base.first_lbp < T0].copy()                 # entered funnel
        df = df[df.coverage_start <= T0]                      # covered at T0
        need_end = T0 + H if mode == "train" else T0
        df = df[df.coverage_end >= need_end]                  # covered through outcome (train)
        df = df[df.first_surg.isna() | (df.first_surg >= T0)]  # not already operated (late stages kept)
        df["index_date"] = T0
        if mode == "train":
            df["y"] = [int(any(T0 < d <= T0 + H for d in surg_by.get(t, [])))
                       for t in df.patient_token]
        rows.append(df)
    coh = pd.concat(rows, ignore_index=True)
    coh["row_id"] = np.arange(len(coh))
    return coh


# ------------------------------------------------------------------------- features
def _win(dates, idx, lo):
    return (dates <= idx) if lo is None else ((dates <= idx) & (dates > idx - lo))


def _z68_bmi(code):
    """Approx numeric BMI from an ICD-10 Z68.* band code (Z68.26 -> 26, Z68.41 -> 42)."""
    s = code[4:]
    if s.startswith("1"):
        return 19.0
    return {"41": 42., "42": 47., "43": 55., "44": 65., "45": 70.}.get(
        s, float(s[:2]) if len(s) >= 2 and s[:2].isdigit() else np.nan)


def _features_single(sub: pd.DataFrame, data: dict) -> pd.DataFrame:
    """Features for one snapshot (patient_token unique within sub)."""
    enc, clm, out = data["enc"], data["clm"], data["out"]
    f = pd.DataFrame(index=sub.index)
    f["patient_token"] = sub.patient_token.values
    f["age"] = sub.index_date.dt.year.values - sub.birth_year.values
    f["sex_M"] = (sub.sex == "M").astype(int).values
    f["sex_U"] = (sub.sex == "U").astype(int).values
    obs_start = np.maximum(sub.coverage_start.values, DATA_START.to_datetime64())
    f["lookback_days"] = (sub.index_date.values - obs_start) / np.timedelta64(1, "D")
    f["short_history"] = (f["lookback_days"] < 90).astype(int)
    f["days_since_first_lbp"] = (sub.index_date - sub.first_lbp).dt.days.values

    key = sub[["patient_token", "index_date"]]
    e = enc.merge(key, on="patient_token", how="inner")
    e = e[e.event_date <= e.index_date]

    def ec(cat, lo):
        s = e[e.event_category == cat]
        return s[_win(s.event_date, s.index_date, lo)].groupby("patient_token").size()
    f = f.join(ec("Imaging", Y1).rename("imaging_count_12mo"), on="patient_token")
    f = f.join(ec("LBP Encounter", Y1).rename("lbp_count_12mo"), on="patient_token")
    f = f.join(ec("LBP Encounter", Q3).rename("lbp_count_3mo"), on="patient_token")
    f = f.join(ec("PT Referral", None).rename("pt_referral_prior"), on="patient_token")
    multi = e[e.spine_level == "Multiple"].groupby("patient_token").size()
    f = f.join(multi.rename("_m"), on="patient_token")
    lbp_e = e[e.event_category == "LBP Encounter"].sort_values("event_date")
    f = f.join(lbp_e.groupby("patient_token").facility_id.last().rename("facility_id"),
               on="patient_token")
    # imaging modality (72148=MRI, 72131=CT, 72100=X-ray) -> advanced workup signal
    img = e[e.event_category == "Imaging"]
    ic = img.event_code.astype(str)
    f["mri_flag"] = f.patient_token.isin(
        img[ic == "72148"].patient_token.unique()).astype(int)
    f["advanced_imaging_flag"] = f.patient_token.isin(
        img[ic.isin(["72148", "72131"])].patient_token.unique()).astype(int)
    # LBP recency + escalation velocity (gap between the two most recent LBPs)
    T0 = sub.index_date.iloc[0]
    lg = lbp_e.groupby("patient_token").event_date
    f = f.join((T0 - lg.max()).dt.days.rename("days_since_last_lbp"), on="patient_token")
    f = f.join(lg.apply(lambda s: (s.iloc[-1] - s.iloc[-2]).days if len(s) >= 2 else np.nan)
               .rename("lbp_gap_days"), on="patient_token")

    # --- progression across funnel stages (LBP -> Imaging -> PT Referral) ---
    img_first = img.groupby("patient_token").event_date.min()
    ref_first = e[e.event_category == "PT Referral"].groupby("patient_token").event_date.min()
    flbp = sub.set_index("patient_token").first_lbp
    f["_imgf"] = f.patient_token.map(img_first)
    f["_reff"] = f.patient_token.map(ref_first)
    f["_flbp"] = f.patient_token.map(flbp)
    f["days_lbp_to_imaging"] = (f["_imgf"] - f["_flbp"]).dt.days
    f["days_imaging_to_referral"] = (f["_reff"] - f["_imgf"]).dt.days
    # escalation recency + acceleration (any non-LBP MSK event)
    esc = e[e.event_category.isin(["Imaging", "PT Referral", "Injection", "Surgical Consult"])]
    f = f.join((T0 - esc.groupby("patient_token").event_date.max()).dt.days
               .rename("days_since_last_escalation"), on="patient_token")
    esc90 = esc[esc.event_date > (T0 - pd.Timedelta(days=90))]
    f = f.join(esc90.groupby("patient_token").size().rename("escalation_events_90d"), on="patient_token")
    f["injection_flag"] = f.patient_token.isin(
        e[e.event_category == "Injection"].patient_token.unique()).astype(int)
    f["surgical_consult_flag"] = f.patient_token.isin(
        e[e.event_category == "Surgical Consult"].patient_token.unique()).astype(int)

    c = clm.merge(key, on="patient_token", how="inner")
    c = c[c.service_date <= c.index_date]
    icd = c[c.code_system == "ICD10CM"]
    f["obese_flag"] = f.patient_token.isin(
        icd[icd.code.str.startswith("E66", na=False)].patient_token.unique()).astype(int)
    f["severe_obese_flag"] = f.patient_token.isin(
        icd[icd.code.str.startswith("E6601", na=False)].patient_token.unique()).astype(int)
    f = f.join(icd.assign(p3=icd.code.str[:3]).groupby("patient_token").p3.nunique()
               .rename("comorbidity_count"), on="patient_token")
    # BMI intensity: precise from Z68.* band codes; E66-subcode fallback; else missing
    # (absence of a BMI code != normal weight, so we leave it NaN for HGB to handle).
    z = icd[icd.code.str.startswith("Z68", na=False)]
    f = f.join(z.assign(b=z.code.map(_z68_bmi)).groupby("patient_token").b.max()
               .rename("bmi"), on="patient_token")
    f.loc[f.bmi.isna() & (f.severe_obese_flag == 1), "bmi"] = 40.0
    f.loc[f.bmi.isna() & (f.obese_flag == 1), "bmi"] = 32.0
    for pre, nm in [("E11", "diabetes_flag"), ("F32", "depression_flag"),
                    ("F41", "anxiety_flag"), ("F17", "tobacco_flag"), ("I10", "htn_flag")]:
        f[nm] = f.patient_token.isin(
            icd[icd.code.str.startswith(pre, na=False)].patient_token.unique()).astype(int)

    def cc(mask, lo=Y1):
        s = c[mask]
        return s[_win(s.service_date, s.index_date, lo)].groupby("patient_token").size()
    f = f.join(cc(c.place_of_service == "Emergency Department").rename("ed_visits_12mo"), on="patient_token")
    f = f.join(cc(c.place_of_service == "Inpatient Hospital").rename("inpatient_admits_12mo"), on="patient_token")
    f = f.join(cc(c.provider_specialty == "Pain Management").rename("pain_mgmt_visits_12mo"), on="patient_token")
    f = f.join(cc(c.provider_specialty == "Primary Care").rename("pcp_visits_12mo"), on="patient_token")
    f = f.join(cc((c.code_system == "NDC") & (c.claim_status == "DISPENSED")).rename("rx_count_12mo"), on="patient_token")
    f = f.join(cc(c.provider_specialty == "Physical Therapy", None).rename("pt_dose"), on="patient_token")

    o = out.merge(key, on="patient_token", how="inner")
    o = o[o.timestamp <= o.index_date]
    f = f.join(o.groupby("patient_token").size().rename("prior_outreach_count"), on="patient_token")
    rec = o[o.timestamp > (o.index_date - pd.Timedelta(days=180))]
    f = f.join(rec.groupby("patient_token").size().rename("recent_outreach"), on="patient_token")
    # hard-rule flags consumed by reach() -> 0
    f["opt_out_flag"] = f.patient_token.isin(
        o[o.attempt_result.isin(["Opted Out", "Unsubscribed"])].patient_token.unique()).astype(int)
    # a decline within 180d blocks outreach unless a new MSK event has occurred since
    last_decl = o[o.attempt_result == "Declined"].groupby("patient_token").timestamp.max()
    last_evt = e.groupby("patient_token").event_date.max()           # latest MSK event <= index
    f = f.join(last_decl.rename("_ld"), on="patient_token").join(last_evt.rename("_le"), on="patient_token")
    declined_180d = f["_ld"].notna() & (f["_ld"] > (T0 - pd.Timedelta(days=180)))
    event_since_decline = f["_le"].notna() & (f["_le"] > f["_ld"])
    f["decline_block"] = (declined_180d & ~event_since_decline).astype(int)
    f = f.drop(columns=["_ld", "_le"])
    # enrolled in a program within the last 12 months -> one episode per year
    er = data["enr"].merge(key, on="patient_token", how="inner")
    er = er[er.enrollment_date <= er.index_date]
    banned_er = er[er.enrollment_date > (er.index_date - pd.Timedelta(days=365))]
    f["completion_block"] = f.patient_token.isin(banned_er.patient_token.unique()).astype(int)

    f["imaging_flag"] = (f["imaging_count_12mo"].fillna(0) > 0).astype(int)
    f["spine_multilevel_flag"] = (f["_m"].fillna(0) > 0).astype(int)
    f = f.drop(columns=["_m", "_imgf", "_reff", "_flbp"])
    cnt = ["imaging_count_12mo", "lbp_count_12mo", "lbp_count_3mo", "pt_referral_prior",
           "comorbidity_count", "ed_visits_12mo", "inpatient_admits_12mo",
           "pain_mgmt_visits_12mo", "pcp_visits_12mo", "rx_count_12mo", "pt_dose",
           "prior_outreach_count", "recent_outreach", "escalation_events_90d"]
    f[cnt] = f[cnt].fillna(0)
    for c in ["days_since_last_lbp", "lbp_gap_days", "days_lbp_to_imaging",
              "days_imaging_to_referral", "days_since_last_escalation"]:
        f[c] = f[c].fillna(3650)
    # funnel stage ordinal (0 LBP .. 4 consult), breadth, and progression velocity
    f["max_stage"] = np.select(
        [f.surgical_consult_flag == 1, f.injection_flag == 1,
         f.pt_referral_prior > 0, f.imaging_flag == 1],
        [4, 3, 2, 1], default=0)
    f["n_stages"] = (1 + f.imaging_flag + (f.pt_referral_prior > 0).astype(int)
                     + f.injection_flag + f.surgical_consult_flag)
    f["progression_rate"] = f["n_stages"] / (f["days_since_first_lbp"] + 30) * 365

    # 12-month counts -> per-year rates, normalized by the observed fraction of the
    # window (floored denom) so counts are comparable across ragged histories.
    # obs_window is that fraction in [0,1] and is itself a feature (protects calibration).
    f["obs_window"] = np.minimum(f.lookback_days.values, 365) / 365.0
    obs_win = np.clip(f["obs_window"].values, 0.25, 1.0)   # floored denom for rates
    f["ed_rate"] = f.ed_visits_12mo.values / obs_win
    f["inpatient_rate"] = f.inpatient_admits_12mo.values / obs_win
    f["pain_mgmt_rate"] = f.pain_mgmt_visits_12mo.values / obs_win
    f["rx_rate"] = f.rx_count_12mo.values / obs_win
    f["imaging_rate"] = f.imaging_count_12mo.values / obs_win
    return f


def build_features(coh: pd.DataFrame, data: dict) -> pd.DataFrame:
    parts = [_features_single(sub, data) for _, sub in coh.groupby("index_date")]
    f = pd.concat(parts).loc[coh.index]      # realign to cohort row order
    return f.reset_index(drop=True)


def encode_facility(f, y=None, fmap=None, global_rate=None):
    """Target-encode facility_id. Fit (fmap None) uses y; else apply saved map."""
    if fmap is None:
        gr = float(np.mean(y))
        agg = pd.DataFrame({"facility_id": f.facility_id.values, "y": y}) \
            .groupby("facility_id").y.agg(["sum", "count"])
        a = 20.0
        fmap = ((agg["sum"] + a * gr) / (agg["count"] + a)).to_dict()
        global_rate = gr
    rate = f.facility_id.map(fmap).fillna(global_rate).values
    return rate, fmap, global_rate


# ------------------------------------------------------------- composite score parts
def divertibility(f):
    # funnel stage weight: earlier = more divertible; surgical-consult patients sink.
    stage = np.ones(len(f))                                      # LBP only
    stage = np.where(f.imaging_flag == 1, 0.9, stage)            # imaging (diagnostic step)
    stage = np.where(f.pt_referral_prior > 0, 0.8, stage)
    stage = np.where(f.injection_flag == 1, 0.4, stage)
    stage = np.where(f.surgical_consult_flag == 1, 0.05, stage)  # on the surgical path
    recency = np.clip(np.exp(-f.days_since_first_lbp / 365.0), 0.3, 1.0)
    return stage * recency


def reach(f):
    """Outreach eligibility x contact-fatigue. Hard rules force reach=0: opted out
    (ever); declined in the last 180 days with no new MSK event since; or enrolled in a
    program in the last 12 months. Otherwise a soft discount for heavy recent contact."""
    r = 1.0 / (1.0 + f.recent_outreach.values)
    blocked = ((f.opt_out_flag.values == 1) | (f.decline_block.values == 1)
               | (f.completion_block.values == 1))
    return np.where(blocked, 0.0, r)


def acuity_high(f):
    return ((f.imaging_flag == 1) | (f.lbp_count_3mo >= 2) |
            (f.pain_mgmt_visits_12mo > 0)).astype(int).values


# --------------------------------------------------------------------------- helpers
def _gain_table(y, p, bins=10):
    order = np.argsort(-p); y = np.asarray(y)[order]; n = len(y); tot = y.sum()
    out = []
    for i in range(1, bins + 1):
        hi = int(round(n * i / bins)); cum = y[:hi].sum()
        out.append((i * 10, hi, int(cum), cum / tot, (cum / hi) / (tot / n)))
    return pd.DataFrame(out, columns=["top_%", "n", "surgeries", "capture", "lift"])


def _calib_metrics(y, p, bins=10):
    """ECE + calibration slope/intercept (the primary metrics for a ranking model).
    Equal-frequency bins; equal-width bins flatter a rare, low-spread outcome by
    over-weighting the dense near-zero region."""
    y, p = np.asarray(y), np.asarray(p)
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    ece = sum(m.mean() * abs(p[m].mean() - y[m].mean())
              for b in np.unique(q) for m in [q == b])
    eps = 1e-6
    lg = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    lr = LogisticRegression().fit(lg, y)
    return ece, lr.coef_[0, 0], lr.intercept_[0]


# ------------------------------------------------------------------------- risk model
def _models():
    return {
        "logistic": Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler()),
                              ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "hgb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400,
                                              l2_regularization=1.0, early_stopping=True,
                                              random_state=0),
    }


def train():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    data = load_data()
    coh = build_cohort(data, "train")
    npat = coh.patient_token.nunique()
    print(f"Cohort: {len(coh):,} snapshot-rows, {npat:,} patients, "
          f"{coh.y.sum():,} positives ({coh.y.mean()*100:.1f}%)")

    feats = build_features(coh, data)
    groups = coh.patient_token.values
    y = coh.y.values
    h = coh.patient_token.apply(lambda t: int(t[:8], 16) % 100).values
    test_mask = h < 15                                   # patient-level split

    X = feats[CLINICAL_FEATURES].astype(float)
    Xdev, ydev, gdev = X[~test_mask], y[~test_mask], groups[~test_mask]
    Xte, yte = X[test_mask], y[test_mask]

    # model selection on patient-grouped CV (AUPRC), then isotonic calibration on OOF
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    best, best_ap, best_name = None, -1, None
    for name, mdl in _models().items():
        oof = cross_val_predict(mdl, Xdev, ydev, cv=sgkf, groups=gdev, method="predict_proba")[:, 1]
        ap = average_precision_score(ydev, oof)
        if ap > best_ap:
            best, best_ap, best_name = mdl, ap, name
    oof = cross_val_predict(best, Xdev, ydev, cv=sgkf, groups=gdev, method="predict_proba")[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, ydev)
    best.fit(Xdev, ydev)
    model = CalibratedModel(best, iso)

    # held-out (patient-disjoint) metrics
    pte = model.predict_proba(Xte)
    ece, _, _ = _calib_metrics(yte, pte)
    lift10 = _gain_table(yte, pte).iloc[0]["lift"]
    print(f"Held-out test (patient-disjoint): ECE {ece:.3f}, AUC {roc_auc_score(yte, pte):.2f}, "
          f"top-decile lift {lift10:.1f}x   [model: {best_name}]")

    # time-disjoint generalization: train 2024 snapshots -> test 2025
    yr = coh.index_date.dt.year.values
    tr, te = yr == 2024, yr >= 2025
    _, fm24, g24 = encode_facility(feats[tr], y[tr])
    Xt = feats.copy()
    Xt["facility_surgery_rate"] = encode_facility(feats, fmap=fm24, global_rate=g24)[0]
    Xt = Xt[CLINICAL_FEATURES].astype(float)
    p25 = clone(best).fit(Xt[tr], y[tr]).predict_proba(Xt[te])[:, 1]
    auc_td = roc_auc_score(y[te], p25)

    # program-effect diagnostic (IPW); confounded -> not used for assignment
    delta = estimate_delta(coh, X, feats, y, data)
    conf = "confounded, all segments negative" if delta and all(v < 0 for v in delta.values()) else "see artifact"
    print(f"Generalization: time-disjoint AUC {auc_td:.2f};  program-effect Δ (IPW) {conf} -> diagnostic only")

    with open(ARTIFACTS_DIR / "model.pkl", "wb") as fh:
        pickle.dump({"model": model, "delta": delta,
                     "features": CLINICAL_FEATURES}, fh)
    print(f"\nSaved -> {ARTIFACTS_DIR/'model.pkl'}")


def estimate_delta(coh, X, feats, y, data) -> dict:
    enr = data["enr"]
    X = X.fillna(X.median())                     # propensity LR can't take NaN (bmi)
    obese = feats.obese_flag.values.astype(bool)
    ac = acuity_high(feats)
    delta = {}
    for prog in ["PT", "WM"]:
        e = enr[enr.program == prog].merge(coh[["row_id", "patient_token", "index_date"]], on="patient_token")
        trows = e[(e.enrollment_date > e.index_date) &
                  (e.enrollment_date <= e.index_date + H)].row_id.unique()
        treated = coh.row_id.isin(trows).values
        elig = obese if prog == "WM" else np.ones(len(coh), bool)
        if treated[elig].sum() < 20:
            continue
        ps = LogisticRegression(max_iter=2000)
        ps.fit(X[elig], treated[elig])
        e_hat = np.clip(ps.predict_proba(X[elig])[:, 1], 0.05, 0.95)
        t, yy, ah = treated[elig], y[elig], ac[elig]
        w = np.where(t, 1 / e_hat, 1 / (1 - e_hat))
        for a in [0, 1]:
            m = ah == a
            if (t & m).sum() < 10 or ((~t) & m).sum() < 10:
                continue
            rt = np.average(yy[t & m], weights=w[t & m])
            rc = np.average(yy[(~t) & m], weights=w[(~t) & m])
            delta[(prog, int(a))] = float(rc - rt)
    return delta


# ----------------------------------------------------------------------------- score
def score():
    art = pickle.load(open(ARTIFACTS_DIR / "model.pkl", "rb"))
    data = load_data()
    coh = build_cohort(data, "score")
    print(f"Scoring cohort: {len(coh):,} patients (T0={SCORE_T0.date()})")
    feats = build_features(coh, data)
    p = art["model"].predict_proba(feats[CLINICAL_FEATURES].astype(float))
    div, rch = divertibility(feats), reach(feats)
    comp = p * div * rch   # calibrated risk x divertibility x reach
    obese = feats.obese_flag.values.astype(bool)
    severe = feats.severe_obese_flag.values.astype(bool)
    ac = acuity_high(feats)

    # Clinical indication rule: PT = first-line for mechanical LBP; WM where obesity
    # is the dominant, addressable driver and the spine is not acutely escalating.
    prog = []
    for i in range(len(feats)):
        if not obese[i]:
            pg = "PT"                       # WM not indicated
        elif severe[i] or ac[i] == 0:
            pg = "WM"                        # obesity-dominant / non-acute
        else:
            pg = "PT"                        # obese but acute spine -> PT first
        prog.append(pg)

    # rescale composite to a 0-100 priority index (order unchanged); top patient = 100
    mx = comp.max()
    priority = (100.0 * comp / mx) if mx > 0 else comp
    out = pd.DataFrame({"patient_token": coh.patient_token.values,
                        "program_recommended": prog,
                        "score": np.round(priority, 2)})
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    OUT_DIR.mkdir(exist_ok=True)
    validate_targets(out)
    out.to_csv(OUT_DIR / "targets.csv", index=False)
    vc = out.program_recommended.value_counts()
    print(f"Wrote {OUT_DIR/'targets.csv'} -> {len(out):,} rows (patient_token, program_recommended, score)")
    print(f"Program split: PT {vc.get('PT', 0):,} / WM {vc.get('WM', 0):,}")
    print("Top 5:")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "score"])
    cmd = ap.parse_args().command
    train() if cmd == "train" else score()
