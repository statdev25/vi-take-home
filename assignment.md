# Applied/Staff Data Scientist — Take-Home Assignment

**Vi Labs — Health Systems & Care Navigation Squad**

---

## The situation

Read `client-brief.md` first. It describes the client, their problem, and what they have asked for.

The short version: Meridian Health Partners wants to identify patients heading toward lumbar spine surgery and get them into conservative care first. They run programs for physical therapy and weight management, and their team has been calling, texting and emailing patients to enroll them. They want Vi to optimize those enrollment campaigns. They have given us three years of history and asked us what to do next quarter.

## What we are asking you to produce

**1. A ranked outreach list.** Which patients Meridian should contact next quarter, in priority order, with a recommended program for each, and a defensible answer to how far down the list to go.

**2. A repository that produces it.** Start from the scaffold provided by Vi. In the README include a brief summary of your methodology with the audience being Vi's DS team. Cover your modeling approach, evaluation strategy, and a pilot study design that Vi would hypothetically propose to the client.

**3. A presentation and discussion,** about an hour in total. Described below.

## Submission format

We ship a scaffold repository. Clone it and build inside it. It contains a `uv` project, a `main.py` with the two entry points already wired, a writer and validator for the output file, a test setup, and a README template. It contains no data-handling code — reading and understanding the files is part of the assignment.

Your submission must satisfy the contract the scaffold sets up. 

```bash
uv run main.py train
uv run main.py score
```

- `out/targets.csv` — `patient_token, program_recommended, score`
- Runs end to end on CPU.
- Dependencies declared in `pyproject.toml`

We are not prescribing a metric. Choosing how to evaluate this is part of the problem.

## The data

Five files, plus a data dictionary. Roughly 30,000 patients over 36 months, extracted 2026-07-01. Data goes under `data/`.

| File | Source |
|---|---|
| `meridian_health_patients.csv` | Meridian's data warehouse |
| `meridian_health_encounters.csv` | Meridian's orthopedic service line |
| `meridian_health_outreach.csv` | Meridian's campaign activity |
| `meridian_health_enrollments.csv` | Meridian's program enrollment records |
| `data_web_claims.csv.gz` | The Vi data web |

## What we are evaluating

**Role: Applied Data Scientist**

- **Study design.** How well you turn the business problem into a pilot that will show Vi's value.
- **Commercial reasoning.** Is the list optimized for what Meridian actually cares about?
- **Presentation skills.** Will a skeptical non-technical stakeholder find your approach and results compelling.

**Role: Staff Data Scientist**

- **Robust methodology.** Did you select a modeling approach that aligns with the value propositions of Vi's product?
- **Generalizable approach.** If Vi signs another client for a similar use case, how much of your solution could be re-used? How much would be reworked from scratch?
- **Architecture & engineering.** Is you solution designed such that it can easily be automated and applied to larger datasets?

**All Roles**

- **Code and pipeline quality.** Is the repository well structured and tested, and does it run end to end as specified?
- **Communication.** Can you demonstrate that you understand how your solution works, and explain why you chose it.

A defensible v1 with a clear thought process and a clean implementation is an excellent submission.

## How much to build

We would rather see a few decisions made well than a lot of components built quickly. So that you know where the bar is:

- **A simple, well-calibrated risk model is sufficient.** We are not scoring on performance metrics or your ability to build a complex model.
- **A single overall estimate of the campaign's effect is sufficient.** Break it down further if you think the breakdown matters, but that is not expected.
- **A modest feature set is expected.** Exhaustive feature engineering is not rewarded, and neither is hyperparameter tuning. We are more interested in everything around the model than the model itself.

## Time

We built this to take somewhere around **8 focused hours**. That is a design target, not a rule, and we are not timing you — but it should tell you what depth we are expecting and where to stop.

If you are well past that, stop and write up what you have. An unfinished component you can explain is worth more to us than a complete one you cannot.

## The presentation

About an hour in total, in three parts.

**20–25 minutes — present your results.** Vi panelists will role-play Meridian stakeholders responsible for operationalizing your target list into an outreach campaign. They are not familiar with machine learning, and they are unsure how Vi's product helps them meet their enrollment targets or Meridian's cost savings goals. Be ready to defend the design in your README — Meridian asked how they will know whether it worked, and they will want to hear it in plain language. This portion is particularly important for the **Applied Data Scientist** role.

Slides are optional. Bring them if they help you.

**20–25 minutes — approach discussion.** The role-play stops and the Vi data science team talks with you about your design, methodology, and implementation approach. This portion is particularly important for the **Staff Data Scientist** role.

**The remaining time is yours.** Bring questions about Vi, the squad, or the work.

## Submitting

Vi will send you a zip archive of the assignment materials and project scaffold. We ask that you make a public github repository with your submission. 

Send an email with a link to the github repository to **tony.tornaritis@vi.co** and **eddie.pantridge@vi.co** when prior to the presentation date in your scheduling email.

Questions of any kind are welcome, whether about the brief, the data, the deliverables, logistics or scheduling. Send them to the same addresses.

---

## Closing notes

You are welcome to use any tools, including AI agents, to prepare your submission. We ask that you mention which tools you used in your submission's README.

The hardest parts of this assignment are not modeling problems, and several of the decisions you have to make do not have a single correct answer. We are more interested in which decisions you noticed you were making than in which way you went on any one of them.