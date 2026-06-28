# Provenance Guard

A backend system for classifying submitted creative content as human- or AI-authored, scoring confidence, surfacing transparency labels, and handling creator appeals.

---

## Detection Signals

The production pipeline uses an ensemble of 3 distinct signals:

1. **LLM Attribution Signal (`llm_score`)**
	- Measures: holistic semantic and stylistic cues of AI authorship.
	- Captures: globally coherent, generic, over-smoothed AI style patterns.
	- Misses: very short text, heavily human-edited AI output, and certain formal human writing styles.

2. **Stylometric Signal (`stylo_score`)**
	- Measures: sentence variance, type-token ratio, punctuation density, and sentence-length regularity.
	- Captures: structural uniformity commonly found in model-generated prose.
	- Misses: poetry, minimalist writing, and short-form text where statistics are weak.

3. **Pattern Reuse Signal (`pattern_score`)**
	- Measures: repeated bigrams, sentence opener reuse, and transition-marker density.
	- Captures: repetitive connective phrasing and cadence often seen in generated drafts.
	- Misses: deliberate rhetorical repetition and formulaic domain writing.

---

## Confidence Scoring

### Ensemble weighting

```text
combined_confidence = 0.5 * llm_score + 0.3 * stylo_score + 0.2 * pattern_score
```

### Threshold mapping

- `0.00 - 0.49` => `likely_human`
- `0.50 - 0.79` => `uncertain`
- `0.80 - 1.00` => `likely_ai`

### Validation approach

The scoring pipeline was validated with multiple samples spanning clear AI-like writing, casual human writing, short/borderline submissions, and formal technical prose. The API returns all three individual scores (`llm_score`, `stylo_score`, `pattern_score`) plus combined confidence so disagreements between signals are visible.

---

## Transparency Labels

The system surfaces one of three labels depending on the combined confidence score. The verbatim text for each variant is defined below.

| Variant | Score Range | Label Text |
|---|---|---|
| **Likely AI** | ≥ 0.80 | "Our system's analysis indicates this content was likely generated with the assistance of an AI tool. This label reflects the output of automated detection signals and may not be accurate. If you are the author and believe this is incorrect, you may submit an appeal." |
| **Uncertain** | 0.50 – 0.79 | "Our system could not confidently determine whether this content is human- or AI-generated. No finding has been made against the author. This label reflects measurement uncertainty." |
| **Likely Human** | < 0.50 | "Our system's analysis did not find strong signals of AI authorship in this content. It appears consistent with human-written work." |

---

## Stack

| Component | Tool |
|---|---|
| API framework | Flask |
| Detection signal 1 | Groq (llama-3.3-70b-versatile) — LLM classification |
| Detection signal 2 | Stylometric heuristics — pure Python |
| Rate limiting | Flask-Limiter |
| Audit log | Structured JSON Lines (`audit_log.jsonl`) |

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq API key:

```
GROQ_API_KEY=your_key_here
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/submit` | Submit text for classification |
| `POST` | `/submit_multimodal` | Submit non-text metadata (`image_metadata`) through attribution pipeline |
| `POST` | `/appeal` | Submit an appeal (allowed only for content labeled `likely_ai`) |
| `GET` | `/appeals` | List appeals currently under human review |
| `POST` | `/appeals/<appeal_id>/review` | Human reviewer decision (`approve` or `deny`) |
| `POST` | `/certificate/verify` | Verify creator attestation and issue provenance certificate |
| `GET` | `/log` | Return recent structured audit log entries |
| `GET` | `/analytics` | Return aggregate analytics metrics |
| `GET` | `/dashboard` | Lightweight web dashboard for analytics |

## Rate Limiting

The submission endpoint uses Flask-Limiter with:

- `10 per minute`
- `100 per day`

This is permissive for normal creators submitting original work while still limiting abuse from scripted flooding. The limiter uses in-memory storage (`memory://`) for local development.

Example evidence capture command:

```bash
for i in $(seq 1 12); do
	curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5000/submit \
		-H "Content-Type: application/json" \
		-d '{"text": "This is a test submission for rate limit testing purposes only.", "creator_id": "ratelimit-test"}'
done
```

Expected result: first 10 responses are `200`, then requests exceed limit and return `429`.

## Audit Log Format

Each line in `audit_log.jsonl` is a structured JSON event.

Submission events include:

- `timestamp`
- `content_id`
- `creator_id`
- `attribution`
- `confidence` (combined score)
- `llm_score`
- `stylo_score`
- `pattern_score`
- `status`

Appeal events include:

- `appeal_id`
- `content_id`
- `creator_id`
- `creator_reasoning`
- `status` (`under_review`)

Appeal review events include:

- `appeal_id`
- `content_id`
- `reviewer_id`
- `decision` (`approve` or `deny`)
- `status` (`approved` or `denied`)

Certificate verification events include:

- `creator_id`
- `certificate_id`
- `status` (`verified`)

## Demo Evidence

### Different confidence outcomes

Observed from local test runs:

```text
high 0.71 uncertain
low  0.377 likely_human
```

### Rate limiting in action

Observed status codes from 12 rapid submissions:

```text
200
200
200
200
200
200
200
200
200
200
429
429
```

### Submission + appeal + review chain in log

Recent audit log entries include the full lifecycle for one content item:

```json
{"event":"submission","content_id":"279d8331-...","attribution":"likely_ai","confidence":0.881,"status":"flagged","timestamp":"..."}
{"event":"appeal","appeal_id":"5e2077ae-...","content_id":"279d8331-...","creator_reasoning":"I wrote this manually.","status":"under_review","timestamp":"..."}
{"event":"appeal_review","appeal_id":"5e2077ae-...","content_id":"279d8331-...","decision":"deny","status":"denied","timestamp":"..."}
```

---

## Stretch Features Implemented

### 1. Ensemble Detection

Implemented 3-signal ensemble scoring with documented weights and conflict handling. When signals disagree, the weighted average naturally shifts borderline content toward `uncertain` rather than forcing a high-confidence verdict.

### 2. Provenance Certificate

Creators can complete verification via `POST /certificate/verify` by submitting an exact attestation phrase:

```text
I certify this submission is my original work.
```

On success, the API issues a `certificate_id`. Future submissions by that creator include a distinguishable `verified_label` inside `provenance_certificate` in the response.

### 3. Analytics Dashboard

`GET /analytics` exposes at least 3 rubric-required metrics:

- detection pattern (`ai_ratio`, `human_ratio`, `uncertain_ratio`)
- appeal rate (`appeal_rate`)
- additional metrics (`avg_confidence`, `pending_appeals`, `reviewed_appeals`, `insufficient_length_rate`)

`GET /dashboard` provides a browser-viewable metrics page.

### 4. Multi-Modal Support

`POST /submit_multimodal` accepts `image_metadata` payloads (`title`, `description`, `tags`) and transforms them into pipeline text for attribution scoring, returning the same structured result schema as text submissions.

---

## Known Limitations

1. Minimalist poetry and short verse can be over-penalized by stylometric and pattern signals because low variance is intentional in those forms.
2. Non-native English writers may produce formal, repetitive constructions that can resemble model output.
3. Short submissions (fewer than 3 sentences or about 20 words) default the stylometric signal to neutral and may reduce confidence quality.

## Spec Reflection

The original plan used a 2-signal weighted model (LLM + stylometric at 0.6/0.4). Implementation diverged by adding a third pattern-reuse signal and shifting to a 0.5/0.3/0.2 ensemble. This change improved explainability for borderline cases by surfacing where repetition-heavy text disagreed with semantic classification.

## AI Usage Notes

1. **Milestone 3 generation assist**
	- Prompted AI to scaffold Flask routes and Groq integration for `POST /submit`.
	- Revision made manually: replaced placeholder output with strict JSON validation and structured audit logging fields required by rubric.

2. **Milestone 4 signal design assist**
	- Prompted AI to generate stylometric metrics and score normalization utilities.
	- Revision made manually: changed thresholds and label language to preserve `uncertain` as non-accusatory and to gate appeals to `likely_ai` only.

3. **Milestone 5 workflow assist**
	- Prompted AI to draft appeal state transitions and queue endpoints.
	- Revision made manually: added `appeal_id`, review endpoint, and immutable log lifecycle (`submission` -> `appeal` -> `appeal_review`) for traceability.
