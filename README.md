# Applied Data Science Take-Home

Begin with [`client-brief.md`](client-brief.md) for the client and their problem, then
[`assignment.md`](assignment.md) for the deliverables, submission contract, and evaluation criteria.

## Setup

```bash
uv sync
```

## Run

```bash
uv run main.py train   # fit and calibrate; writes artifacts/model.pkl
uv run main.py score   # score the 2026-07 cohort; writes out/targets.csv
```

`out/targets.csv` contains exactly `patient_token, program_recommended, score` for all patients, in priority order. The pipeline runs end to end on CPU in under a minute.

## Test

```bash
uv run pytest
```

## Layout
`main.py` — entrypoint. `assignment/` — implementation. `tests/` — tests. `data/` — the extract.

---

# Methodology
*Audience: Vi's data science team. Covers the modeling approach, evaluation strategy, and proposed pilot study design.*

## Problem and approach
Meridian, a Medicare ACO, requires a ranked quarterly list of low-back-pain (LBP) patients to divert from avoidable lumbar spine surgery into physical therapy (PT) or weight management (WM), with priority given to patients early in the clinical journey. Each patient receives one recommended program and a priority score:

**score = P(surgery within 12 months) × divertibility × reachability**

Only the first factor is a model; divertibility and reachability are transparent policy parameters that Meridian controls. The multiplicative form requires a patient to be simultaneously at risk, still divertible, and contactable in order to rank highly.

## Modeling approach
**Cohort — prevalent multi-snapshot.** Patients are observed at quarterly snapshots while in the funnel, and training and scoring use the same snapshot representation to avoid train–serve skew. This is the most consequential design decision: a first-LBP landmark cohort flattens the funnel features (every patient has one LBP visit and no imaging), yielding AUC 0.69, whereas a prevalent snapshot allows the features to vary (AUC 0.74) and matches the live scoring population. All funnel stages are retained — only already-operated patients are excluded — so the model observes late-stage signal; late-stage patients are instead deprioritized in the score. The label is a performed spine surgery within 12 months. The training set comprises 66,361 snapshot-rows across 19,565 patients, with 2,475 positives (3.7%).

**Risk model.** A calibrated `HistGradientBoostingClassifier`, with logistic regression as a baseline, fit on log-loss. Cross-validation uses patient-grouped `StratifiedGroupKFold` so that a recurring patient never spans folds, and probabilities are calibrated with isotonic regression on out-of-fold predictions to prevent leakage. The model uses eight features with no hyperparameter tuning: `days_since_first_lbp, max_stage, age, bmi (ICD Z68), rx_rate, comorbidity_count, escalation_events_90d, obs_window`. Features were pruned from approximately 40 to 8 using both AUC and calibration; `obs_window` contributes negligibly to AUC but is retained because it protects the probabilities.

**Policy parameters.** Divertibility is a funnel-stage weight (LBP 1.0, imaging 0.9, PT referral 0.8, injection 0.4, surgical consult 0.05) multiplied by a recency decay. Reachability applies hard exclusions — opt-out, enrollment within 12 months, and a recent decline with no subsequent flare — and a contact-fatigue discount.

**Program assignment.** Programs are assigned by a clinical indication rule: PT is first-line, and WM is selected where obesity is the dominant modifiable driver. A data-driven program-effect estimate (segment-level IPW Δ) proved confounded, as enrollees are systematically sicker, and is therefore retained only as a diagnostic; the pilot resolves program choice.

**Cutoff.** The depth of the list follows a decision rule rather than a fixed count: contact patients while the marginal net value is positive, which reduces to a break-even hurdle, ARR ≥ (outreach + take-up × program) / surgery cost. Pre-pilot, the list is worked top-down to the limit of call-center capacity.

## Evaluation strategy
Calibration is the primary metric because both the priority ranking and the break-even economics depend on the predicted probabilities being accurate, not merely correctly ordered; discrimination is reported only as a secondary check that risk concentrates at the top of the list:

| Check | Result |
|---|---|
| Calibration (primary): ECE  | 0.004  |
| Discrimination: AUC / top-decile lift | 0.74 / ~3.5× |

The two disjoint splits are chosen to mirror deployment: the patient-disjoint split tests generalization to new members, and the time-disjoint split tests the next-quarter prediction the model will make in practice.

A retrospective check confirms earlier targeting: the top 2,000 patients on Vi's list direct approximately 4% of outreach to injection- or consult-stage patients, compared with approximately 34% in Meridian's historical outreach.

Causal-uplift ranking was considered and rejected. Historical outreach was non-random and had no holdout, so per-patient treatment effects are neither identifiable nor verifiable offline (the observational Δ is confounded). A credible causal estimate requires the pilot.

## Pilot study design
The pilot is a randomized encouragement design. The members are randomized before any outreach into Treatment (Vi outreach) and Control (usual care), stratified on risk decile, program, and obesity × acuity, with cluster randomization by clinic where provider spillover is a concern. The primary outcome, analysis, sample size, and stopping rules are pre-registered. The design is rolling: each patient is randomized once, and Control patients are withheld from every quarterly campaign for the full 12-month window.

- **Primary analysis — Intention-to-Treat (ITT):** everyone offered outreach versus Control, where ARR = P(surgery | control) − P(surgery | treated) and NNT = 1 / ARR. ITT is unbiased and does not condition on enrollment.
- **Secondary analysis — CACE:** ITT ÷ enrollment rate, estimating the effect on patients who enroll. The study is powered on ITT and reports CACE. The pilot also randomizes dual-eligible members across PT and WM to learn, causally, which program diverts more surgery.
- **Sizing and power:** the pilot is powered on the ITT effect, which is diluted by take-up (only some contacted members enroll), so it requires a substantial sample drawn from the top of the list and accumulated across quarters via the rolling design; enrollment take-up is the dominant lever on how large and how long it must run.
- **Guardrails and safety:** pre-specified interim analyses with an early-stopping rule if a large benefit or harm emerges; contamination and crossover are logged and excluded; opt-outs are always honored; and care-manager clinical override is retained throughout.
- **Success criteria (pre-specified):** a statistically significant ARR greater than zero; positive net savings at Meridian's costs; and demonstrably earlier engagement. These determine whether to scale, to refine the policy parameters and refit, or to stop.

The design is ethical under equipoise: outreach is unproven, Control retains the full standard of care, only the new outreach is withheld and only briefly, and Control patients may still enter the programs through usual care.

## Key decisions
- A risk model with simple rules rather than a causal-uplift ranker, which the data do not support.
- A prevalent multi-snapshot cohort with train–serve consistency — the most consequential modeling decision.
- Calibration prioritized over AUC, both as the fitting objective and as the feature-selection criterion.
- Late-funnel patients retained in the model but deprioritized in the score.
- Program assignment by clinical rule
- No assumed dollar figures for costs

## Generalization and tools
The solution is a reusable template. The cohort builder, windowed-aggregation feature layer, calibrated model, score framework, pilot design, and MLOps scaffold transfer directly to a new client; only the clinical event taxonomy, outcome definition, program rules, and source-schema mapping require rework. The pipeline runs on CPU in approximately 30 seconds, and the feature layer ports to SQL or Spark for larger extracts. Developed with Claude Code; implemented in Python (pandas, scikit-learn) and run via uv.
