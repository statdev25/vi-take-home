# Data Dictionary

**Meridian Health Partners — MSK/spine care navigation**
**Extract date:** 2026-07-01 · **Window:** 2023-07-01 through 2026-06-30

Five files, provided under `data/`. Roughly 30,000 patients. ALL DATA IS SYNTHETIC -- NOTHING TAKEN FROM REAL WORLD DATASETS.

`patient_token` is the join key across all five files. All patient identifiers are tokenized; there is no PII in the extract and no crosswalk.

Four files come from Meridian's own systems. `data_web_claims.csv.gz` is a Vi asset — a longitudinal claims feed maintained by Vi's data engineers.

---

## `meridian_health_patients.csv`

**Source:** Meridian's data warehouse
**Grain:** one row per patient

| Column | Type | Description |
|---|---|---|
| `patient_token` | string | Tokenized patient identifier. Primary key |
| `birth_year` | integer | Year of birth |
| `sex` | string | `F`, `M`, `U` |
| `zip3` | string | First three digits of the patient's ZIP code |
| `pcp_id` | string | Identifier for the patient's attributed primary care physician. Null where the patient is not attributed |
| `coverage_start` | date | Start of the patient's coverage period under Meridian's accountable-care arrangement |
| `coverage_end` | date | End of the patient's coverage period |

---

## `meridian_health_encounters.csv`

**Source:** Meridian's orthopedic service line, EHR-derived data warehouse
**Grain:** one row per clinical event

A curated extract of musculoskeletal events. Broader than office visits — it includes imaging, injections, referrals and surgeries alongside encounters. It is not a complete record of the patient's care.

| Column | Type | Description |
|---|---|---|
| `patient_token` | string | Foreign key to `meridian_health_patients.csv` |
| `event_id` | string | Unique identifier for the event |
| `event_date` | date | Date of the event |
| `event_category` | string | `LBP Encounter`, `Imaging`, `PT Referral`, `Injection`, `Surgical Consult`, `Spine Surgery` |
| `event_code` | string | Clinical code for the event. ICD-10-CM or CPT |
| `code_system` | string | `ICD10CM`, `CPT` |
| `spine_level` | string | Anatomical level, where recorded. `L3-L4`, `L4-L5`, `L5-S1`, `Multiple` |
| `event_status` | string | Status of the event as recorded in the source system. `Performed`, `Scheduled`, `Cancelled`, `No Show` |
| `facility_id` | string | Identifier for the facility where the event occurred |

---

## `meridian_health_outreach.csv`

**Source:** Meridian's campaign activity export
**Grain:** one row per contact attempt

Historical activity from Meridian's enrollment campaigns for the physical therapy and weight management programs.

| Column | Type | Description |
|---|---|---|
| `patient_token` | string | Foreign key to `meridian_health_patients.csv` |
| `timestamp` | timestamp | Date and time of the attempt, in local time |
| `message` | string | Codename of the campaign creative used for this attempt. `Back in Motion`, `Stay in the Game`, `Stand Tall`, `Fresh Start`, `More Moments` |
| `program` | string | Program the creative's call to action asks the patient to enroll in. `PT` (physical therapy), `WM` (weight management) |
| `channel` | string | `Phone`, `SMS`, `Email` |
| `attempt_result` | string | Outcome of the attempt as recorded by the platform. `Connected`, `Voicemail`, `No Answer`, `Invalid Number`, `Declined`, `Delivered`, `Failed`, `Opted Out`, `Bounced`, `Opened`, `Unsubscribed` |

---

## `meridian_health_enrollments.csv`

**Source:** Meridian's program enrollment records: its physical therapy program and its weight management vendor
**Grain:** one row per program enrollment

Every enrollment in either program, however the patient was referred.

| Column | Type | Description |
|---|---|---|
| `patient_token` | string | Foreign key to `meridian_health_patients.csv` |
| `enrollment_id` | string | Unique identifier for the enrollment |
| `program` | string | Program enrolled in. `PT` (physical therapy), `WM` (weight management) |
| `enrollment_date` | date | Date of the first session |

---

## `data_web_claims.csv.gz`

**Source:** The Vi data web
**Grain:** one row per claim line

Longitudinal medical and pharmacy claims. Broad rather than MSK-specific — this is the general-purpose feature source, and it is not where Meridian's clinical events live.

| Column | Type | Description |
|---|---|---|
| `patient_token` | string | Foreign key to `meridian_health_patients.csv` |
| `claim_line_id` | string | Unique identifier for the claim line |
| `service_date` | date | Date of service |
| `place_of_service` | string | `Office`, `Outpatient Hospital`, `Emergency Department`, `Inpatient Hospital`, `Telehealth`, `Skilled Nursing Facility` |
| `provider_specialty` | string | Specialty of the rendering provider. `Orthopedic Surgery`, `Neurosurgery`, `Physical Therapy`, `Pain Management`, `Primary Care`, `Rheumatology`, `Radiology`, `Emergency Medicine`, `Anesthesiology`, `Chiropractic`, `Cardiology`, `Nephrology`, `Endocrinology`, `Behavioral Health`, `Other` |
| `code_system` | string | `ICD10CM`, `CPT`, `HCPCS`, `NDC` |
| `code` | string | Clinical code, in the vocabulary given by `code_system` |
| `claim_status` | string | Adjudication status of the claim line. `PAID`, `DENIED`, `DISPENSED`, `REJECTED`, `REVERSED` |
