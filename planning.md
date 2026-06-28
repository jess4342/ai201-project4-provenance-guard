# Provenance Guard — Planning Document

## Architecture Narrative

A piece of submitted text travels through Provenance Guard as follows:

1. **Client** sends `POST /submit` with the raw text and an author identifier.
2. **Flask API** receives the request and fans the text out to two independent detection signals in sequence.
3. **Signal 1 — LLM Classifier** sends the text to Groq (llama-3.3-70b-versatile) with a structured prompt asking whether the passage reads as human- or AI-generated. The model returns a probability of AI authorship as a float between 0 and 1.
4. **Signal 2 — Stylometric Analyzer** computes four measurable statistical properties of the text entirely in pure Python: sentence-length variance, type-token ratio (vocabulary diversity), punctuation density, and average sentence length. These raw measurements are normalized and combined into a single stylometric score between 0 and 1.
5. **Confidence Scorer** merges the two signal scores into a single combined confidence score using a weighted average (60 % LLM, 40 % stylometric), producing a float in [0, 1] where values approaching 1 indicate stronger AI-authorship signal.
6. **Label Generator** maps the combined score to one of three human-readable transparency labels based on fixed thresholds.
7. **Audit Logger** writes the full record — submission ID, author ID, both raw signal scores, the combined confidence score, the assigned label, and a UTC timestamp — to a SQLite database.
8. **Response** returns the submission ID, combined confidence score, and label text to the caller.

For the **appeal flow**, the author (and only the author) sends `POST /appeal` with their submission ID and a justification of up to 200 characters. Appeals are accepted only when the submission label is `Likely AI`. The API validates authorship, verifies the current label/state is appeal-eligible, updates the submission status from `flagged` to `under_review`, logs the appeal, and returns an appeal ID. A human reviewer accesses `GET /appeals` to see the queue — each entry shows the original text, both signal scores, the combined confidence, the assigned label, and the author's justification — then calls `POST /appeals/<id>/review` with an `approve` or `deny` decision to close the appeal.

---

## Architecture

```
SUBMISSION FLOW
===============

 Client
   │
   │  POST /submit
   │  { text, author_id }
   │
   ▼
 Flask API
   │
   ├────────────────────────────────────────────────────┐
   │  raw text                                          │  raw text
   ▼                                                    ▼
 LLM Classifier (Groq)                     Stylometric Analyzer (pure Python)
   llama-3.3-70b-versatile                   • sentence-length variance
   prompt: "score AI authorship 0–1"         • type-token ratio
   output: llm_score (float 0–1)             • punctuation density
                │                             • avg sentence length
                │ llm_score                   output: stylo_score (float 0–1)
                │                                        │
                └──────────────┬─────────────────────────┘
                               │ llm_score, stylo_score
                               ▼
                        Confidence Scorer
                          combined = 0.6 * llm_score
                                   + 0.4 * stylo_score
                          output: confidence (float 0–1)
                               │
                               │ confidence
                               ▼
                        Label Generator
                          < 0.50  → "Likely Human"
                          0.50–0.80 → "Uncertain"
                          ≥ 0.80  → "Likely AI"
                               │
                               │ label text, confidence
                               ▼
                        Audit Logger (SQLite)
                          writes: submission_id, author_id,
                                  llm_score, stylo_score,
                                  confidence, label, timestamp
                               │
                               ▼
                        JSON Response
                          { submission_id, confidence, label }


APPEAL FLOW
===========

 Author (must match submission author_id)
   │
   │  POST /appeal
   │  { submission_id, author_id, justification (≤ 200 chars) }
   │
   ▼
 Flask API
   ├── validate: author_id matches submission record
   ├── validate: justification length ≤ 200 chars
   ├── validate: submission status == "flagged"
   │
   ▼
 Audit Logger (SQLite)
   updates status: "flagged" → "under_review"
   writes: appeal_id, submission_id, justification, timestamp
   │
   ▼
 JSON Response { appeal_id, status: "under_review" }


REVIEW FLOW (human reviewer)
=============================

 Reviewer
   │
   │  GET /appeals
   │  returns: [ { submission_id, text, llm_score, stylo_score,
   │               confidence, label, justification } ]
   │
   │  POST /appeals/<appeal_id>/review
   │  { decision: "approve" | "deny" }
   │
   ▼
 Audit Logger (SQLite)
   updates status: "approved" | "denied"
   writes: reviewer_decision, reviewed_at timestamp
   │
   ▼
 JSON Response { appeal_id, status }
```

The submission flow is a single synchronous path: text → two independent signals → score → label → log → response. The appeal flow is a two-step async process: the author opens an appeal (status → `under_review`) and a separate reviewer closes it (status → `approved` or `denied`).

---

## Detection Signals

### Signal 1 — LLM-Based Classification (Groq)

**What it measures:** Semantic and stylistic coherence holistically. The model evaluates whether the text's word choices, sentence constructions, topic transitions, and overall register match patterns it associates with AI-generated versus human-generated writing.

**Why it differs between human and AI writing:** Large language models generate text by predicting the most probable next token, which tends to produce globally coherent, grammatically clean, topically on-topic prose with consistent register. Human writing exhibits more idiosyncratic phrasing, unexpected digressions, emotional register shifts, and structural irregularity that models have learned to distinguish.

**Output format:** A single float in [0, 1] — the model's estimated probability that the text is AI-authored. Elicited via a structured prompt that asks the model to respond with only a number.

**Blind spots:**
- AI text that has been lightly paraphrased or deliberately degraded by a human will likely score lower than it should.
- Short texts (< 50 tokens) give the model insufficient signal and produce noisy scores.
- Highly formulaic human writing (legal boilerplate, standardized form letters) may score high because it resembles AI output structurally.
- The model's own biases about "what AI sounds like" may penalize certain non-native English writing styles.

---

### Signal 2 — Stylometric Heuristics (Pure Python)

**What it measures:** Four statistical surface properties of the text:

| Property | Measurement | AI tendency |
|---|---|---|
| Sentence-length variance | Standard deviation of word counts per sentence | Low variance (uniform lengths) |
| Type-token ratio (TTR) | `unique_words / total_words` | Higher TTR (broader vocabulary) but less colloquial |
| Punctuation density | `punctuation_chars / total_chars` | Lower density (cleaner, less expressive punctuation) |
| Average sentence length | Mean word count across sentences | Moderate and consistent |

Each property is normalized to [0, 1] and combined into a single `stylo_score`. Higher scores indicate more AI-like statistical patterns.

**Why it differs between human and AI writing:** AI text generation optimizes for readability and coherence, which produces statistically uniform output: sentences cluster around a mean length, vocabulary is broad but consistent, and punctuation is used correctly but sparingly. Human writers are more variable — they use run-on sentences, fragments, em-dashes, ellipses, repetition, and unusual word choices that make their statistical fingerprints noisier.

**Output format:** A single float in [0, 1] computed entirely in pure Python with no external dependencies.

**Blind spots:**
- Poetry, song lyrics, and highly structured literary forms (villanelles, sonnets) exhibit low variance by design and will score as AI-like.
- Writers with minimalist styles (Hemingway-esque short declarative sentences) will produce low variance scores.
- Very short texts (< 3 sentences) produce statistically unreliable measurements — the heuristics need enough sentences to compute meaningful variance.
- Domain-specific technical writing (instructions, API docs) naturally exhibits low punctuation density and uniform sentence length.

---

### Signal Combination

```
combined_score = 0.6 * llm_score + 0.4 * stylo_score
```

The LLM signal carries more weight (60 %) because it captures semantic properties that the surface-level heuristics cannot. The stylometric signal (40 %) adds a structurally independent check and provides a hedge against prompt injection or model hallucination on the classification task. Both signals must pull in the same direction to push the combined score above the 0.80 "Likely AI" threshold, which reduces false positives.

---

## Uncertainty Representation

### Threshold Table

| Combined Score | Label | Interpretation |
|---|---|---|
| 0.00 – 0.49 | Likely Human | Signals do not support AI authorship |
| 0.50 – 0.79 | Uncertain | Signals conflict or are insufficient to decide |
| 0.80 – 1.00 | Likely AI | Both signals strongly indicate AI authorship |

### What intermediate scores mean

A score of **0.62** means the signals are present but not aligned: the LLM may have detected some AI-like phrasing while the stylometric properties look more human (or vice versa). The system does not have enough confidence to make a definitive attribution. The "Uncertain" label is the honest representation of that state — it is not a finding against the author.

A score of **0.45** means both signals weakly favor human authorship. The system returns "Likely Human" and does not flag the submission.

A score of **0.88** means both signals strongly agree that the text is AI-generated. At this threshold, the system flags the submission and surface the "Likely AI" label.

### Why these thresholds

The upper threshold is deliberately high at 0.80 (rather than the symmetric 0.67) to minimize false positives. Incorrectly labeling a human writer's work as AI-generated is the highest-cost error — it harms reputation, undermines trust, and triggers appeals. Erring toward "Uncertain" rather than "Likely AI" on ambiguous cases is the appropriate conservative choice. The wide uncertain band (0.50–0.80) captures exactly these ambiguous cases and surfaces them for human review via the appeals flow rather than making an automated false accusation.

### Score calibration approach

- LLM score: elicited by asking the model to return only a float 0–1 in its response, parsed directly.
- Stylometric score: each of the four properties is min-max normalized against expected ranges for the English language (sentence variance 0–15 words, TTR 0–1, punctuation density 0–0.2, avg sentence length 5–40 words), then averaged into a single score.
- Combined score: weighted average as defined above, clamped to [0, 1].

---

## Transparency Label Design

The three label variants are designed to be honest, non-accusatory, and informative. They are written for a general audience, not a technical one.

### Likely AI (combined_score ≥ 0.80)

> "Our system's analysis indicates this content was likely generated with the assistance of an AI tool. This label reflects the output of automated detection signals and may not be accurate. If you are the author and believe this is incorrect, you may submit an appeal."

### Uncertain (0.50 ≤ combined_score < 0.80)

> "Our system could not confidently determine whether this content is human- or AI-generated. No finding has been made against the author. This label reflects measurement uncertainty."

### Likely Human (combined_score < 0.50)

> "Our system's analysis did not find strong signals of AI authorship in this content. It appears consistent with human-written work."

---

## Appeals Workflow

### Who can submit an appeal

Only the original submitting author, identified by `author_id` matching the `author_id` stored in the submission record, and only when the submission is labeled `Likely AI`. Third-party appeals are not accepted.

### What the author provides

- `submission_id`: the ID of the submission being appealed
- `author_id`: must match the stored submission author
- `justification`: a plain-text string, maximum **200 characters**, explaining why the author believes the classification is incorrect (e.g., "This is an original poem I wrote by hand. My writing style uses short declarative sentences.")

The 200-character cap keeps the justification focused and prevents the field from becoming a circumvention vector. The author does not need to prove authorship technically — the justification is a human-readable statement of context for the reviewer.

### System behavior on receipt

1. Validate `author_id` matches submission record — return 403 if not.
2. Validate `justification` ≤ 200 characters — return 400 if not.
3. Validate submission status is `flagged` (not already `under_review`, `approved`, or `denied`) — return 409 if not.
4. Update submission status: `flagged` → `under_review`.
5. Write appeal record to audit log: `appeal_id`, `submission_id`, `justification`, `appealed_at` timestamp.
6. Return `{ appeal_id, status: "under_review" }`.

### What the human reviewer sees

The reviewer calls `GET /appeals` and receives a list of all `under_review` appeals. Each entry contains:

- **Submission text** (full original content)
- **llm_score** and **stylo_score** (the two raw signal values)
- **combined_score** and **label** (what the system decided)
- **Author justification** (the ≤ 200 character statement)
- **appeal_id** and **submission_id**
- **Timestamps** for original submission and appeal

The reviewer then calls `POST /appeals/<appeal_id>/review` with `{ "decision": "approve" }` (author's appeal accepted — label updated or removed) or `{ "decision": "deny" }` (classification upheld). Either decision is written to the audit log with a `reviewed_at` timestamp. The audit trail is immutable — original scores and labels are never deleted, only status is updated.

---

## Anticipated Edge Cases

### 1. Minimalist poetry or structured verse

**Scenario:** A human author submits a haiku, a minimalist prose poem, or a piece with deliberate anaphora (heavy repetition). These forms intentionally use simple vocabulary, short uniform sentences, and low punctuation density — exactly the statistical fingerprint the stylometric signal associates with AI text. The stylometric score will be artificially high.

**Mitigation:** The combined score weights LLM classification at 60 %. If the LLM recognizes the content as a genuine creative form, the combined score may stay below 0.80. However, very short pieces also give the LLM insufficient context, so the score may still land in the Uncertain band. The wide uncertain band (0.50–0.80) exists precisely to catch these cases — the system surfaces uncertainty rather than a false accusation.

### 2. Lightly edited AI-generated text

**Scenario:** A user generates text with an AI tool, then edits it substantially — rewrites some sentences, adds personal anecdotes, inserts punctuation errors, changes vocabulary. The final text is a blend. The LLM may detect residual AI patterns in the unedited sections, but the stylometric signal may read the overall text as human-like due to added variance.

**Mitigation:** The two signals will likely disagree, keeping the combined score in the Uncertain band rather than triggering a "Likely AI" label. This is an intentional design trade-off: the system should not confidently accuse when signals conflict. If the LLM score alone is above 0.80 but stylometric is below 0.50, the combined score (0.6 × 0.80 + 0.4 × 0.50 = 0.68) correctly lands in Uncertain.

### 3. Very short submissions (< 3 sentences)

**Scenario:** A user submits a one-sentence caption or a two-line bio. The stylometric analyzer needs multiple sentences to compute meaningful variance. The LLM has minimal context to evaluate.

**Mitigation:** The system should detect text below a minimum length threshold and either return an explicit `insufficient_length` flag in the response alongside a reduced-confidence score, or default the stylometric component to 0.5 (neutral) rather than a computed value.

### 4. Non-native English writing

**Scenario:** A human author writing in English as a second language may produce text with unusually uniform sentence structures, limited vocabulary diversity, and reduced punctuation — characteristics that overlap with AI output signatures. The LLM may also have been trained on primarily native-English examples of "human writing."

**Mitigation:** This is a known systemic blind spot. The label design explicitly states results "may not be accurate" and directs authors to the appeals flow. There is no automated fix for this — it is a limitation to document and monitor.

---

## AI Tool Plan

### M3 — Submission Endpoint + First Signal

**Spec sections to provide:** Detection Signals (Signal 1), Architecture diagram (submission flow), Uncertainty Representation (score format and output structure).

**What to request:** Flask app skeleton with `POST /submit` endpoint, SQLite schema creation, and the `classify_with_llm(text) -> float` function using the Groq client. The function should send a structured prompt asking for a single float response and parse it.

**Verification:** Call the function directly in a Python REPL with three inputs — a clearly AI-sounding paragraph, a clearly human casual message, and a borderline academic text. Confirm the three outputs differ meaningfully before wiring into the endpoint.

---

### M4 — Second Signal + Confidence Scoring

**Spec sections to provide:** Detection Signals (Signal 2 property table and normalization approach), Uncertainty Representation (combination formula and thresholds), Architecture diagram.

**What to request:** `compute_stylometric_score(text) -> float` function implementing all four properties with normalization, plus a `combine_scores(llm_score, stylo_score) -> dict` function returning `{ "confidence": float, "label": str }`.

**Verification:** Run the stylometric function on the same three test inputs from M3. Confirm that clearly AI text (uniform, clean) scores higher than clearly human text (variable, casual). Then run the full scoring pipeline end-to-end and check that score ranges map to the correct label variants.

---

### M5 — Production Layer (Labels + Appeals)

**Spec sections to provide:** Transparency Label Design (exact label text for all three variants), Appeals Workflow (full state machine, validation rules, reviewer payload), Architecture diagram (appeal and review flows).

**What to request:** Label generation logic integrated into the response, `POST /appeal` endpoint with authorship validation and 200-char enforcement, `GET /appeals` endpoint returning full reviewer payload, `POST /appeals/<id>/review` endpoint with status update and audit logging.

**Verification:** 
- Hit the submit endpoint and confirm all three label variants are reachable by feeding inputs that produce scores in each range.
- Submit an appeal as the correct author, confirm status changes to `under_review`.
- Attempt an appeal with a mismatched `author_id`, confirm 403.
- Attempt an appeal with a justification > 200 chars, confirm 400.
- Approve and deny appeals, confirm status updates and audit log entries.
