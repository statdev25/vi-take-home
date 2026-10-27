# Meridian Health Partners — Client Brief

**Prepared by:** Vi Labs, Health Systems & Care Navigation
**For:** Applied Data Scientist engagement, MSK/spine care navigation
**Date of brief:** 2026-07-15
**Data extract:** 2026-07-01

---

## 1. Who the client is

Meridian Health Partners is a multispecialty provider group of roughly 400 physicians across a mid-Atlantic metro area and its surrounding counties. They run primary care, an orthopedic service line, imaging, and a handful of ambulatory surgical sites.

Meridian is a value-based care provider participating in a Medicare accountable-care arrangement. They are obligated to pay the total cost of care for the members attributed to them. **Every patient in the dataset you receive is covered under that arrangement.**

## 2. The clinical problem

Lumbar spine is one of Meridian's large cost categories. The patient pathway is well understood: a patient presents with low back pain, gets imaged, gets referred, may get injections, sees a surgeon, and somewhere along that sequence the decision to undergo an expensive surgery is made.

Conservative care — structured physical therapy, and weight management where obesity is a contributing factor — can be effective at preventing the need for surgery. Meridian believes that a meaningful share of patients who end up in the OR could have been diverted if their team had reached them earlier and enrolled them in a PT or weight management program.

## 3. What Meridian has been doing

Meridian has been running enrollment campaigns for the two programs for about two years, contacting patients by phone, SMS, and email. The campaigns have never been evaluated. There was no holdout, no randomization, and no pre-registered outcome.

Historical campaign activity is included in the extract.

## 4. The ask, in Meridian's words

> "Give us a ranked list of patients to contact next quarter, tell me which program each one should be offered, tell me how far down the list to go, and tell me how we'll know whether it worked. It's great if Vi can help drive additional enrollments into our programs, but ideally Vi's list will flag members that would benefit from these programs earlier in the patient journey compared to our existing outreach because we believe an earlier enrollment will increase the likelihood that surgery can be avoided."

---

## 5. What you are receiving

Five files, covering 2023-07-01 through 2026-06-30, extracted 2026-07-01. Approximately 30,000 patients. See `data-dictionary.md` for column-level descriptions.

| File | Source |
|---|---|
| `meridian_health_patients.csv` | Meridian's enterprise data warehouse |
| `meridian_health_encounters.csv` | Meridian's orthopedic service line, EHR-derived |
| `meridian_health_outreach.csv` | Meridian's campaign activity |
| `meridian_health_enrollments.csv` | Meridian's program enrollment records |
| `data_web_claims.csv.gz` | The Vi data web |

The first four come from Meridian's own systems and arrive as they arrive. `data_web_claims.csv.gz` is a Vi asset — a longitudinal claims feed maintained by another team and harmonized before it reaches an engagement. It is the broad feature source; it is not MSK-specific and it is not where Meridian's clinical events live.

All patient identifiers are tokenized. There is no PII in the extract and no crosswalk.
