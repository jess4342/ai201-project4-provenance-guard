import json
import os
import random
import re
import statistics
import string
import uuid
from collections import Counter
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq


LOG_PATH = "audit_log.jsonl"
CERT_CHALLENGE_STORE = {}
VERIFIED_CERT_BY_CREATOR = {}

LIKELY_AI_LABEL = (
    "Our system's analysis indicates this content was likely generated with the assistance "
    "of an AI tool. This label reflects the output of automated detection signals and "
    "may not be accurate. If you are the author and believe this is incorrect, you may "
    "submit an appeal."
)
UNCERTAIN_LABEL = (
    "Our system could not confidently determine whether this content is human- or "
    "AI-generated. No finding has been made against the author. This label reflects "
    "measurement uncertainty."
)
LIKELY_HUMAN_LABEL = (
    "Our system's analysis did not find strong signals of AI authorship in this content. "
    "It appears consistent with human-written work."
)


load_dotenv()
_groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=_groq_key) if _groq_key else None

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(
        error="Too many requests. Please slow down and try again.",
        status=429,
    ), 429

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def clamp_score(value):
    return max(0.0, min(1.0, value))


def parse_score(raw_text):
    if raw_text is None:
        return 0.5
    match = re.search(r"\d*\.?\d+", str(raw_text))
    if not match:
        return 0.5
    try:
        return clamp_score(float(match.group(0)))
    except ValueError:
        return 0.5


def split_sentences(text):
    return [chunk.strip() for chunk in re.split(r"[.!?]+", text or "") if chunk and chunk.strip()]


def tokenize_words(text):
    return re.findall(r"\b[a-zA-Z']+\b", (text or "").lower())


def normalize(value, min_value, max_value):
    if max_value <= min_value:
        return 0.0
    return clamp_score((value - min_value) / (max_value - min_value))


def log_event(entry):
    record = dict(entry)
    record["timestamp"] = utc_now_iso()
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def read_log(limit=20):
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []
    events = []
    for line in lines[-max(0, int(limit)):]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def read_all_events():
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def compute_stylometric_score(text):
    sentences = split_sentences(text)
    words = tokenize_words(text)

    if len(sentences) < 3 or len(words) < 20:
        return {
            "stylo_score": 0.5,
            "insufficient_length": True,
            "components": {
                "sentence_variance_ai": 0.5,
                "type_token_ratio_ai": 0.5,
                "punctuation_density_ai": 0.5,
                "average_sentence_length_ai": 0.5,
            },
        }

    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences if sentence.strip()]
    sentence_variance = statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0

    unique_words = len(set(words))
    type_token_ratio = unique_words / max(1, len(words))

    punctuation_chars = sum(1 for ch in (text or "") if ch in string.punctuation)
    punctuation_density = punctuation_chars / max(1, len(text or ""))

    average_sentence_length = statistics.mean(sentence_lengths) if sentence_lengths else 0.0

    sentence_variance_ai = 1.0 - normalize(sentence_variance, 0.0, 15.0)
    type_token_ratio_ai = normalize(type_token_ratio, 0.35, 0.75)
    punctuation_density_ai = 1.0 - normalize(punctuation_density, 0.00, 0.20)
    average_sentence_length_ai = normalize(average_sentence_length, 5.0, 40.0)

    score = clamp_score(
        (
            sentence_variance_ai
            + type_token_ratio_ai
            + punctuation_density_ai
            + average_sentence_length_ai
        )
        / 4.0
    )

    return {
        "stylo_score": score,
        "insufficient_length": False,
        "components": {
            "sentence_variance_ai": sentence_variance_ai,
            "type_token_ratio_ai": type_token_ratio_ai,
            "punctuation_density_ai": punctuation_density_ai,
            "average_sentence_length_ai": average_sentence_length_ai,
        },
        "raw": {
            "sentence_variance": sentence_variance,
            "type_token_ratio": type_token_ratio,
            "punctuation_density": punctuation_density,
            "average_sentence_length": average_sentence_length,
        },
    }

def compute_structural_regularity_score(text):
    sentences = split_sentences(text)
    lowered = (text or "").lower()
    words = tokenize_words(lowered)
    paragraphs = [p.strip() for p in (text or "").split("\n") if p.strip()]

    if len(sentences) < 3 or len(words) < 20:
        return {
            "structure_score": 0.5,
            "components": {}
        }

    # 1. Sentence rhythm (Coefficient of Variation)
    sentence_lengths = [len(tokenize_words(s)) for s in sentences if s.strip()]
    mean_sl = statistics.mean(sentence_lengths) if sentence_lengths else 0.0
    std_sl = statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    cv_sl = std_sl / mean_sl if mean_sl > 0 else 0
    # Low variance (e.g. CV of 0.2) = AI; High variance (0.6+) = Human
    rhythm_ai = 1.0 - normalize(cv_sl, 0.2, 0.6)

    # 2 & 7. Opening / Template repetition (first 3 words)
    openings = [" ".join(tokenize_words(s.lower())[:3]) for s in sentences if len(tokenize_words(s)) >= 3]
    opening_counts = Counter(openings)
    repeated_openings = sum(count for op, count in opening_counts.items() if count > 1)
    opening_ai = normalize(repeated_openings / max(1, len(sentences)), 0.0, 0.4)

    # 3. Paragraph symmetry (length variance)
    para_lengths = [len(tokenize_words(p)) for p in paragraphs if p]
    mean_pl = statistics.mean(para_lengths) if para_lengths else 0.0
    std_pl = statistics.pstdev(para_lengths) if len(para_lengths) > 1 else 0.0
    cv_pl = std_pl / mean_pl if mean_pl > 0 else 0
    para_sym_ai = 1.0 - normalize(cv_pl, 0.1, 0.5)

    # 4. Lexical recycling (Repeated Trigrams)
    trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    trigram_counts = Counter(trigrams)
    repeated_trigrams = sum(count for tg, count in trigram_counts.items() if count > 1)
    recycling_ai = normalize(repeated_trigrams / max(1, len(trigrams)), 0.0, 0.15)

    # 5. Hedging density
    hedges = {"may", "might", "can", "could", "generally", "typically", "often"}
    hedge_hits = sum(1 for w in words if w in hedges) + lowered.count("in many cases")
    hedging_ai = normalize(hedge_hits / max(1, len(words)), 0.0, 0.05)

    # 6. Transition overload
    transitions = {"first", "second", "third", "finally", "therefore", "thus", "furthermore", "moreover", "additionally"}
    trans_hits = sum(1 for w in words if w in transitions)
    trans_ai = normalize(trans_hits / max(1, len(sentences)), 0.0, 0.5)

    structure_score = clamp_score(
        (rhythm_ai + opening_ai + para_sym_ai + recycling_ai + hedging_ai + trans_ai) / 6.0
    )

    return {
        "structure_score": structure_score,
        "components": {
            "rhythm_ai": rhythm_ai,
            "opening_ai": opening_ai,
            "para_sym_ai": para_sym_ai,
            "recycling_ai": recycling_ai,
            "hedging_ai": hedging_ai,
            "trans_ai": trans_ai
        }
    }

def classify_with_llm(text):
    fallback = compute_structural_regularity_score(text)["structure_score"]
    if not groq_client:
        return fallback

    prompt = (
        "You are classifying text for AI authorship likelihood. "
        "Return only a single float from 0 to 1 where 1 means very likely AI-generated.\n\n"
        f"Text:\n{text}\n"
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content if response.choices else None
        return parse_score(content)
    except Exception:
        return fallback


def combine_ensemble_scores(llm_score, stylo_score, pattern_score):
    llm_score = clamp_score(float(llm_score))
    stylo_score = clamp_score(float(stylo_score))
    pattern_score = clamp_score(float(pattern_score))

    # Keep pattern signal low-impact; rely primarily on semantic + stylometric agreement.
    confidence = clamp_score((0.65 * llm_score) + (0.30 * stylo_score) + (0.05 * pattern_score))

    # Strong and moderate consensus guards to reduce obvious-AI false negatives.
    if llm_score >= 0.80 and stylo_score >= 0.70:
      confidence = max(confidence, 0.82)
    if llm_score >= 0.74 and stylo_score >= 0.70:
      confidence = max(confidence, 0.80)

    if confidence >= 0.80:
        attribution = "likely_ai"
    elif confidence >= 0.50:
        attribution = "uncertain"
    else:
        attribution = "likely_human"

    return {
        "confidence": confidence,
        "attribution": attribution,
    }


def label_for_attribution(attribution):
    if attribution == "likely_ai":
        return LIKELY_AI_LABEL
    if attribution == "likely_human":
        return LIKELY_HUMAN_LABEL
    return UNCERTAIN_LABEL


def get_latest_submission(content_id):
    events = read_all_events()
    latest = None
    for event in events:
        if event.get("event_type") != "submission":
            continue
        if event.get("content_id") == content_id:
            latest = event
    return latest


def get_pending_appeals():
    appeals = {}
    for event in read_all_events():
        if event.get("event_type") == "appeal":
            appeals[event["appeal_id"]] = dict(event)
        elif event.get("event_type") == "appeal_review":
            appeal_id = event.get("appeal_id")
            if appeal_id in appeals:
                appeals[appeal_id]["status"] = event.get("status")
                appeals[appeal_id]["reviewed_at"] = event.get("timestamp")

    pending = [a for a in appeals.values() if a.get("status") == "under_review"]
    pending.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return pending


def get_creator_submissions(creator_id, limit=25):
    if not creator_id:
        return []

    events = read_all_events()

    # map appeals by content_id
    appeal_status_by_content = {}
    for e in events:
        if e.get("event_type") == "appeal":
            cid = e.get("content_id")
            if cid not in appeal_status_by_content:
                appeal_status_by_content[cid] = "under_review"
        elif e.get("event_type") == "appeal_review":
            # find appeal event first
            for a in events:
                if a.get("event_type") == "appeal" and a.get("appeal_id") == e.get("appeal_id"):
                    cid = a.get("content_id")
                    appeal_status_by_content[cid] = e.get("status")

    rows = []
    for event in reversed(events):
        if event.get("event_type") != "submission":
            continue
        if event.get("creator_id") != creator_id:
            continue

        cid = event.get("content_id")

        # human verification
        cert = VERIFIED_CERT_BY_CREATOR.get(creator_id, {}).get("verified", False)

        appeal_status = appeal_status_by_content.get(cid)

        if appeal_status == "approved":
            final_state = "repealed"
        elif appeal_status == "under_review":
            final_state = "in_review"
        elif event.get("attribution") == "likely_ai":
            final_state = "flagged"
        else:
            final_state = "clear"

        preview = (event.get("text") or "").strip().replace("\n", " ")

        rows.append({
            "content_id": cid,
            "timestamp": event.get("timestamp"),
            "attribution": event.get("attribution"),
            "confidence": event.get("confidence"),
            "preview": preview[:180],
            "appeal_status": appeal_status or "none",
            "final_state": final_state,
            "human_verified": cert,
        })

        if len(rows) >= limit:
            break

    return rows

def build_submission_result(text, creator_id, content_type="text", metadata=None):
    llm_score = classify_with_llm(text)
    stylo = compute_stylometric_score(text)
    structure = compute_structural_regularity_score(text)
    combined = combine_ensemble_scores(llm_score, stylo["stylo_score"], structure["structure_score"])

    content_id = str(uuid.uuid4())
    attribution = combined["attribution"]
    confidence = combined["confidence"]
    label = label_for_attribution(attribution)
    status = "flagged" if attribution == "likely_ai" else "clear"

    cert = VERIFIED_CERT_BY_CREATOR.get(creator_id, {"verified": False})
    result = {
        "content_id": content_id,
        "creator_id": creator_id,
        "content_type": content_type,
        "attribution": attribution,
        "confidence": confidence,
        "label": label,
        "llm_score": llm_score,
        "stylo_score": stylo["stylo_score"],
        "structure_score": structure["structure_score"],
        "status": status,
        "provenance_certificate": cert,
    }

    if metadata:
        result["metadata"] = metadata

    log_event(
        {
            "event_type": "submission",
            "content_id": content_id,
            "creator_id": creator_id,
            "text": text,
            "content_type": content_type,
            "metadata": metadata or {},
            "attribution": attribution,
            "confidence": confidence,
            "label": label,
            "llm_score": llm_score,
            "stylo_score": stylo["stylo_score"],
            "structure_score": structure["structure_score"],
            "status": status,
            "appeal_filed": False,
            "provenance_certificate": cert,
        }
    )
    return result


@app.route("/assets/human_verif.png", methods=["GET"])
def badge_asset():
    return send_from_directory(os.getcwd(), "human_verif.png")


@app.route("/assets/PG_favicon.png", methods=["GET"])
def favicon_asset():
    return send_from_directory(os.getcwd(), "PG_favicon.png")


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = (data.get("creator_id") or "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400
    if not creator_id:
        return jsonify({"error": "creator_id is required"}), 400

    result = build_submission_result(text=text, creator_id=creator_id)
    return jsonify(result), 200


@app.route("/submit_multimodal", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit_multimodal():
    data = request.get_json(silent=True) or {}
    creator_id = (data.get("creator_id") or "").strip()
    content_type = (data.get("content_type") or "image_metadata").strip()
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    tags = data.get("tags") or []

    if not creator_id:
        return jsonify({"error": "creator_id is required"}), 400
    if not title and not description:
        return jsonify({"error": "title or description is required"}), 400
    if not isinstance(tags, list):
        return jsonify({"error": "tags must be a list"}), 400

    text = " ".join([title, description, " ".join([str(t) for t in tags])]).strip()
    metadata = {
        "title": title,
        "description": description,
        "tags": tags,
    }
    result = build_submission_result(
        text=text,
        creator_id=creator_id,
        content_type=content_type,
        metadata=metadata,
    )
    return jsonify(result), 200


@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = (data.get("content_id") or "").strip()
    creator_id = (data.get("creator_id") or "").strip()
    creator_reasoning = (data.get("creator_reasoning") or "").strip()

    if not content_id:
        return jsonify({"error": "content_id is required"}), 400
    if not creator_id:
        return jsonify({"error": "creator_id is required"}), 400
    if not creator_reasoning:
        return jsonify({"error": "creator_reasoning is required"}), 400
    if len(creator_reasoning) > 200:
        return jsonify({"error": "creator_reasoning must be <= 200 chars"}), 400

    submission = get_latest_submission(content_id)
    if not submission:
        return jsonify({"error": "content_id not found"}), 404

    if submission.get("creator_id") != creator_id:
        return jsonify({"error": "creator_id mismatch"}), 403

    if submission.get("attribution") != "likely_ai":
        return jsonify({"error": "appeals are only allowed for likely_ai decisions"}), 409

    for pending in get_pending_appeals():
        if pending.get("content_id") == content_id:
            return jsonify({"error": "an appeal is already under review"}), 409

    appeal_id = str(uuid.uuid4())
    entry = {
        "event_type": "appeal",
        "appeal_id": appeal_id,
        "content_id": content_id,
        "creator_id": creator_id,
        "creator_reasoning": creator_reasoning,
        "status": "under_review",
        "appeal_filed": True,
        "text": submission.get("text", ""),
        "llm_score": submission.get("llm_score"),
        "stylo_score": submission.get("stylo_score"),
        "structure_score": submission.get("structure_score"),
        "confidence": submission.get("confidence"),
        "label": submission.get("label"),
        "attribution": submission.get("attribution"),
        "submitted_at": submission.get("timestamp"),
    }
    log_event(entry)

    return (
        jsonify(
            {
                "appeal_id": appeal_id,
                "content_id": content_id,
                "status": "under_review",
                "message": "Appeal received and queued for human review.",
            }
        ),
        200,
    )


@app.route("/appeals", methods=["GET"])
def appeals_list():
    pending = get_pending_appeals()
    payload = []
    for item in pending:
        payload.append(
            {
                "appeal_id": item.get("appeal_id"),
                "content_id": item.get("content_id"),
                "creator_id": item.get("creator_id"),
                "creator_reasoning": item.get("creator_reasoning"),
                "appealed_at": item.get("timestamp"),
                "text": item.get("text", ""),
                "llm_score": item.get("llm_score"),
                "stylo_score": item.get("stylo_score"),
                "pattern_score": item.get("pattern_score"),
                "confidence": item.get("confidence"),
                "label": item.get("label"),
                "attribution": item.get("attribution"),
                "submitted_at": item.get("submitted_at"),
                "status": "under_review",
            }
        )
    return jsonify({"appeals": payload}), 200


@app.route("/appeals/<appeal_id>/review", methods=["POST"])
def review_appeal(appeal_id):
    data = request.get_json(silent=True) or {}
    decision = (data.get("decision") or "").strip().lower()
    reviewer_id = (data.get("reviewer_id") or "").strip()

    if decision not in {"approve", "deny"}:
        return jsonify({"error": "decision must be approve or deny"}), 400
    if not reviewer_id:
        return jsonify({"error": "reviewer_id is required"}), 400

    pending = {item.get("appeal_id"): item for item in get_pending_appeals()}
    if appeal_id not in pending:
        return jsonify({"error": "appeal not found or already closed"}), 404

    status = "approved" if decision == "approve" else "denied"
    log_event(
        {
            "event_type": "appeal_review",
            "appeal_id": appeal_id,
            "content_id": pending[appeal_id].get("content_id"),
            "decision": decision,
            "status": status,
            "reviewer_id": reviewer_id,
        }
    )

    return jsonify({"appeal_id": appeal_id, "status": status}), 200


@app.route("/creator_submissions", methods=["GET"])
def creator_submissions():
    creator_id = (request.args.get("creator_id") or "").strip()

    if not creator_id:
        return jsonify({"error": "creator_id is required"}), 400

    return jsonify(
        {"submissions": get_creator_submissions(creator_id)}
    ), 200


@app.route("/certificate/challenge", methods=["POST"])
def certificate_challenge():
    data = request.get_json(silent=True) or {}
    creator_id = (data.get("creator_id") or "").strip()
    content_id = (data.get("content_id") or "").strip()
    paragraph_hint = (data.get("paragraph_hint") or "").strip()

    if not creator_id:
        return jsonify({"error": "creator_id is required"}), 400
    if not content_id:
      return jsonify({"error": "content_id is required"}), 400

    submission = get_latest_submission(content_id)
    if not submission:
      return jsonify({"error": "content_id not found"}), 404
    if submission.get("creator_id") != creator_id:
      return jsonify({"error": "content_id does not belong to creator_id"}), 403

    source_text = (submission.get("text") or "").strip()
    excerpt = source_text[:200]

    challenge_token = str(uuid.uuid4())
    question_bank = [
      f"For submission {content_id[:8]}, describe one revision you made to improve the argument in this excerpt: '{excerpt}'.",
      f"For submission {content_id[:8]}, what sentence from this excerpt did you rewrite and why: '{excerpt}'?",
      f"For submission {content_id[:8]}, which claim in this excerpt needed stronger evidence, and what change did you make: '{excerpt}'?",
    ]
    challenge_question = random.choice(question_bank)

    CERT_CHALLENGE_STORE[challenge_token] = {
        "creator_id": creator_id,
        "content_id": content_id,
        "paragraph_hint": paragraph_hint,
        "excerpt": excerpt,
        "question": challenge_question,
        "issued_at": utc_now_iso(),
    }

    return (
        jsonify(
            {
                "challenge_token": challenge_token,
                "creator_id": creator_id,
                "content_id": content_id,
                "paragraph_hint": paragraph_hint,
                "submission_excerpt": excerpt,
                "challenge_question": challenge_question,
            }
        ),
        200,
    )


@app.route("/certificate/verify", methods=["POST"])
def certificate_verify():
    data = request.get_json(silent=True) or {}
    creator_id = (data.get("creator_id") or "").strip()
    attestation = (data.get("attestation") or "").strip()
    challenge_token = (data.get("challenge_token") or "").strip()
    challenge_response = (data.get("challenge_response") or "").strip()
    evidence = data.get("evidence") or {}

    if not creator_id:
        return jsonify({"error": "creator_id is required"}), 400
    if not attestation:
        return jsonify({"error": "attestation is required"}), 400
    if not challenge_token or challenge_token not in CERT_CHALLENGE_STORE:
        return jsonify({"error": "valid challenge_token is required"}), 400

    challenge = CERT_CHALLENGE_STORE[challenge_token]
    if challenge.get("creator_id") != creator_id:
        return jsonify({"error": "challenge_token does not belong to creator_id"}), 403

    active_minutes = int(evidence.get("active_minutes", 0) or 0)
    edit_sessions = int(evidence.get("edit_sessions", 0) or 0)
    major_revisions = int(evidence.get("major_revisions", 0) or 0)
    max_paste_chars = int(evidence.get("max_paste_chars", 0) or 0)
    revision_notes_count = int(evidence.get("revision_notes_count", 0) or 0)

    checks = {
        "attestation_present": bool(attestation),
        "challenge_response_present": len(challenge_response) >= 40,
        "active_minutes_ok": active_minutes >= 20,
        "edit_sessions_ok": edit_sessions >= 40,
        "major_revisions_ok": major_revisions >= 2,
        "max_paste_ok": max_paste_chars <= 300,
        "revision_notes_ok": revision_notes_count >= 1,
    }
    verified = all(checks.values())

    certificate = {
        "verified": verified,
        "creator_id": creator_id,
        "content_id": challenge.get("content_id"),
        "certificate_id": str(uuid.uuid4()) if verified else None,
        "badge_image_url": "/assets/human_verif.png" if verified else None,
        "checks": checks,
    }

    if verified:
        VERIFIED_CERT_BY_CREATOR[creator_id] = certificate

    log_event(
        {
            "event_type": "certificate_verify",
            "creator_id": creator_id,
            "verified": verified,
            "challenge_token": challenge_token,
            "checks": checks,
        }
    )

    return jsonify(certificate), 200


@app.route("/analytics", methods=["GET"])
def analytics():
    events = read_all_events()

    submissions = [e for e in events if e.get("event_type") == "submission"]
    appeals = [e for e in events if e.get("event_type") == "appeal"]
    reviews = [e for e in events if e.get("event_type") == "appeal_review"]
    certs = [e for e in events if e.get("event_type") == "certificate_verify" and e.get("verified")]

    likely_ai = sum(1 for e in submissions if e.get("attribution") == "likely_ai")
    uncertain = sum(1 for e in submissions if e.get("attribution") == "uncertain")
    likely_human = sum(1 for e in submissions if e.get("attribution") == "likely_human")

    reviewed = len(reviews)
    pending = len(get_pending_appeals())

    return (
        jsonify(
            {
                "total_submissions": len(submissions),
                "likely_ai": likely_ai,
                "uncertain": uncertain,
                "likely_human": likely_human,
                "total_appeals": len(appeals),
                "appeals_reviewed": reviewed,
                "appeals_pending": pending,
                "verified_certificates": len(certs),
            }
        ),
        200,
    )


@app.route("/log", methods=["GET"])
def get_log():
    limit = request.args.get("limit", default=20, type=int)
    return jsonify(read_log(limit=max(1, min(limit, 500)))), 200


@app.route("/dashboard", methods=["GET"])
def dashboard():
    html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>PG Dashboard</title>
      <link rel="icon" type="image/png" href="/assets/PG_favicon.png" />
      <style>
        body { margin:0; font-family:Segoe UI,Tahoma,sans-serif; background:#eff9f4; color:#123a2d; }
        .wrap { max-width: 980px; margin: 18px auto; padding: 0 14px; }
        .hero { border:1px solid #cde8dd; border-radius:14px; background:#fff; padding:14px; }
        .hero h1 { margin:0; }
        .grid { margin-top:12px; display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; }
        .card { border:1px solid #d8ede4; border-radius:10px; background:#fff; padding:10px; }
        .k { color:#5a7c70; font-size:.8rem; text-transform:uppercase; }
        .v { font-size:1.3rem; font-weight:700; margin-top:4px; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="hero">
          <h1>Provenance Guard Analytics</h1>
          <div style="color:#5a7c70; margin-top:6px;">Live metrics, refreshed every 5s.</div>
        </div>
        <div id="metrics" class="grid"></div>
      </div>
      <script>
        const labels = {
          total_submissions: 'Submissions',
          likely_ai: 'Likely AI',
          uncertain: 'Uncertain',
          likely_human: 'Likely Human',
          total_appeals: 'Total Appeals',
          appeals_reviewed: 'Appeals Reviewed',
          appeals_pending: 'Appeals Pending',
          verified_certificates: 'Verified Certs'
        };

        async function refresh() {
          const res = await fetch('/analytics');
          const data = await res.json();
          const root = document.getElementById('metrics');
          root.innerHTML = Object.entries(labels)
            .map(([key, label]) => `<div class="card"><div class="k">${label}</div><div class="v">${data[key] ?? 0}</div></div>`)
            .join('');
        }

        refresh();
        setInterval(refresh, 5000);
      </script>
    </body>
    </html>
    """
    return html


@app.route("/", methods=["GET"])
def home():
    html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Provenance Guard</title>
      <link rel="icon" type="image/png" href="/assets/PG_favicon.png" />
      <style>
        :root {
          --bg: #eff9f4;
          --surface: #ffffff;
          --ink: #133a2d;
          --muted: #5f7e73;
          --edge: #cde8dd;
          --accent: #1f9d75;
          --accent-2: #0f766e;
          --danger: #dc2626;
          --warn: #ca8a04;
          --ok: #15803d;
          --shadow: 0 6px 20px rgba(18, 90, 62, 0.10);
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          color: var(--ink);
          font-family: Segoe UI, Tahoma, sans-serif;
          background: radial-gradient(circle at 0% 0%, #f7fffb 0, #eaf7f0 55%, #e3f1ea 100%);
        }
        .shell { max-width: 1240px; margin: 16px auto; padding: 0 16px 20px; }
        .hero { border: 1px solid var(--edge); border-radius: 16px; padding: 16px; background: linear-gradient(130deg, #ffffff, #f2fbf7); box-shadow: var(--shadow); margin-bottom: 12px; }
        .hero h1 { margin: 0; font-size: 1.9rem; }
        .sub { margin-top: 6px; color: var(--muted); }
        .toolbar { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
        .chip { border: 1px solid var(--edge); border-radius: 999px; padding: 6px 11px; background: #fff; font-size: 0.82rem; color: var(--muted); }
        .layout-switch { border: 1px solid var(--edge); border-radius: 8px; background: #fff; padding: 6px 10px; color: var(--ink); }
        .nav { margin-top: 12px; border: 1px solid var(--edge); border-radius: 12px; background: var(--surface); padding: 9px; box-shadow: var(--shadow); display: flex; flex-wrap: wrap; gap: 8px; }
        .nav button { width: auto; text-align: center; border: 1px solid #d7ece3; border-radius: 8px; background: #fff; color: var(--ink); padding: 9px 12px; margin-bottom: 0; cursor: pointer; font-weight: 600; transition: transform .16s ease, background .16s ease; }
        .nav button:hover { transform: translateY(-1px); background: #f5fffb; }
        .nav button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
        .panel { border: 1px solid var(--edge); border-radius: 12px; background: var(--surface); padding: 14px; box-shadow: var(--shadow); opacity: 0; transform: translateY(8px); animation: panelIn .22s ease forwards; }
        @keyframes panelIn { to { opacity: 1; transform: translateY(0); } }
        .section-title { margin: 0 0 10px; font-size: 1.22rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; }
        .card { border: 1px solid #dcefe7; border-radius: 10px; padding: 10px; background: #fcfffd; }
        .row { display: flex; flex-direction: column; gap: 5px; margin-bottom: 8px; }
        label { font-size: 0.84rem; color: var(--muted); }
        input, textarea, select { width: 100%; border: 1px solid #cfe5dc; border-radius: 8px; padding: 8px; background: #fff; color: var(--ink); font: inherit; }
        textarea { min-height: 100px; resize: vertical; }
        .btn { border: 0; border-radius: 8px; background: var(--accent); color: #fff; font-weight: 700; padding: 8px 11px; cursor: pointer; transition: transform .16s ease, filter .16s ease; }
        .btn:hover { transform: translateY(-1px); filter: brightness(.97); }
        .btn.secondary { background: var(--accent-2); }
        .hint { color: var(--muted); font-size: 0.81rem; margin: 2px 0 0; }
        .output { margin-top: 8px; padding: 10px; border-radius: 8px; background: #0f2a23; color: #d8fff0; min-height: 130px; overflow: auto; white-space: pre-wrap; word-break: break-word; font-size: 0.82rem; line-height: 1.35; }
        .output.rate-limit {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        }
        .pretty-result { margin-top: 10px; border: 1px solid #cde9de; border-radius: 12px; background: linear-gradient(130deg, #ffffff, #f0faf5); padding: 12px; }
        .pretty-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; margin-top: 8px; }
        .metric { border: 1px solid #dbefe7; border-radius: 10px; padding: 8px; background: #fff; }
        .metric .k { font-size: 0.73rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
        .metric .v { font-size: 1.06rem; margin-top: 2px; font-weight: 700; }
        .score-high { color: var(--ok); }
        .score-mid { color: var(--warn); }
        .score-low { color: var(--danger); }
        .label-box { margin-top: 8px; border-left: 4px solid var(--accent); padding: 8px 10px; border-radius: 8px; background: #effcf5; color: #2e5547; }
        .badge-wrap { margin-top: 10px; display: flex; align-items: center; gap: 10px; border: 1px solid #cde9de; border-radius: 10px; background: #f0fff7; padding: 8px; }
        .badge-wrap img { width: 72px; height: 72px; object-fit: contain; border-radius: 8px; }
        .json-toggle { margin-top: 8px; display: inline-flex; gap: 8px; align-items: center; }
        .hidden { display: none !important; }
        .access { border: 1px dashed #b6d8c9; border-radius: 9px; padding: 10px; background: #f5fff9; color: #44695d; }
        .pill { display:inline-block; border:1px solid #d7ece3; border-radius:999px; padding:4px 10px; font-size:.78rem; color:#44695d; background:#fff; }
        .review-list { margin-top:8px; max-height:220px; overflow:auto; border:1px solid #d7ece3; border-radius:8px; padding:8px; background:#fff; }
        .review-item { border:1px solid #e1f0ea; border-radius:8px; padding:7px; margin-bottom:7px; cursor:pointer; background:#fafffd; }
        .review-item:hover { background:#f2fbf7; }
        .tile-grid{
            display:grid;
            grid-template-columns:repeat(auto-fill,minmax(330px,1fr));
            gap:20px;
            margin-top:18px;
        }

        .tile-card{
            background:#fff;
            border:1px solid #dbe8e3;
            border-radius:18px;
            padding:18px;
            box-shadow:0 10px 24px rgba(0,0,0,.06);
            transition:.18s;
            display:flex;
            flex-direction:column;
            gap:14px;
        }

        .tile-card:hover{
            transform:translateY(-4px);
            box-shadow:0 18px 36px rgba(0,0,0,.09);
        }

        .tile-header{
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            gap: 12px;
        }

        .tile-id{
            font-weight:700;
            font-size:.9rem;
            color:#20513b;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
            line-height: 1.4;
            word-break: break-word;
        }
        
        .copy-icon {
            cursor: pointer;
            color: #839d91;
            transition: color 0.15s ease, transform 0.1s ease;
        }
        
        .copy-icon:hover {
            color: #1f9d75;
            transform: scale(1.1);
        }

        .tile-card .tile-title {
        font-weight: 700;
        margin-bottom: 6px;
        }

        .tile-card .tile-meta {
        font-size: 0.85rem;
        color: #5f6f68;
        margin-top: 4px;
        }

        .status-pill{
            padding:5px 12px;
            border-radius:999px;
            font-size:.72rem;
            font-weight:700;
            display:inline-block;
        }

        .status-ai{
            background:#fde8e8;
            color:#b42318;
        }

        .status-human{
            background:#e9f9ef;
            color:#067647;
        }

        .status-uncertain{
            background:#fff7d6;
            color:#a15c00;
        }

        .status-repealed{
            background:#dff7ff;
            color:#0b7285;
        }   

        .tile-metrics{
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:10px;
        }

        .metric-box{
            background:#f7fbf8;
            border-radius:10px;
            padding:8px;
        }

        .metric-label{
            font-size:.68rem;
            color:#6b7b74;
            text-transform:uppercase;
        }

        .metric-value{
            font-weight:700;
            margin-top:3px;
        }

        .preview{
            background:#f8faf9;
            border-left:4px solid #27b37e;
            padding:12px;
            border-radius:10px;
            color:#3f4b46;
            line-height:1.45;
        }

        .verify-row{
            display:flex;
            justify-content:space-between;
            align-items:center;
        }

        .verify-row img{
            width:58px;
            height:58px;
        }
        @media (max-width: 920px) { .nav button { flex: 1 1 auto; } }
      </style>
    </head>
    <body>
      <div class="shell">
        <div class="hero">
          <div style="display: flex; align-items: center; gap: 14px;">
            <img src="/assets/PG_favicon.png" alt="PG Logo" style="width: 36px; height: 36px; object-fit: contain; border-radius: 6px;" />
            <h1>Provenance Guard Console</h1>
          </div>
          <div class="sub">Modern creator workflow: submit content, optionally appeal, reviewer triage in a separate role view.</div>
          <div class="toolbar">
            <span class="chip">Role</span>
            <select id="role_select"><option value="submitter" selected>Submitter</option><option value="reviewer">Reviewer</option></select>
            <span class="pill">Submission -> Appeal (if flagged) -> Reviewer Decision</span>
          </div>
          <div class="nav">
            <button class="nav-btn active submitter-only" data-tab="submission_tab">Submission</button>
            <button class="nav-btn submitter-only" data-tab="appeal_tab">Appeal</button>
            <button class="nav-btn reviewer-only" data-tab="review_tab">Reviewer</button>
            <button class="nav-btn submitter-only" data-tab="verify_tab">Human Verify</button>
            <button class="nav-btn submitter-only" data-tab="my_submissions_tab">My Submissions</button>
            <button class="nav-btn reviewer-only" data-tab="log_tab">Audit Log</button>
            <button class="nav-btn reviewer-only" data-tab="analytics_tab">Analytics</button>
          </div>
        </div>

        <div id="layout_root" class="layout">
          <main>
            <section id="submission_tab" class="panel">
              <h2 class="section-title">Submission</h2>
              <div class="card">
                <div class="row">
                  <label>Content Type</label>
                  <select id="submission_mode">
                    <option value="text" selected>Text</option>
                    <option value="image_metadata">Multimodal (Image Metadata)</option>
                  </select>
                </div>
                <div class="row"><label>Creator ID</label><input id="submit_creator" value="demo-author" /></div>
                <div id="text_inputs"><div class="row"><label>Text</label><textarea id="submit_text">I wrote this draft quickly on my train commute, and the phrasing may look uneven but reflects my real writing process.</textarea></div></div>
                <div id="multimodal_inputs" class="hidden">
                  <div class="row"><label>Title</label><input id="mm_title" value="Foggy Alley at Dawn" /></div>
                  <div class="row"><label>Description</label><textarea id="mm_desc">A narrow alley lit by warm storefront lights with early-morning fog and wet pavement reflections.</textarea></div>
                  <div class="row"><label>Tags (comma-separated)</label><input id="mm_tags" value="photo,street,dawn" /></div>
                </div>
                <button class="btn" id="submit_btn">Submit Content</button>
                <div class="hint">If flagged as likely_ai, use returned content_id in Appeal.</div>
                <div id="submit_pretty" class="pretty-result">Submit content to see polished result view.</div>
                <div class="json-toggle"><button class="btn secondary" id="submit_json_toggle">Show Full JSON</button><span class="hint">JSON is hidden by default for cleaner demos.</span></div>
                <pre id="submit_out" class="output hidden">Waiting...</pre>
              </div>
            </section>

            <section id="appeal_tab" class="panel hidden">
              <h2 class="section-title">Appeal (Submitter)</h2>
              <div class="card">
                <div class="row"><label>Creator ID</label><input id="appeal_creator" value="demo-author" /></div>
                <button class="btn secondary" id="appeal_load_btn" style="margin-bottom:8px;">Load Flagged Submissions</button>
                <div class="row"><label>Content ID</label><select id="appeal_content_id"><option value="">Load submissions first...</option></select></div>
                <div class="row"><label>Reasoning (max 200 chars)</label><textarea id="appeal_reason" maxlength="200">I wrote this piece myself. My style is formal and may resemble generated text.</textarea></div>
                <button class="btn secondary" id="appeal_btn">Submit Appeal</button>
                <div class="hint">Appeals are only available for likely_ai decisions.</div>
                <pre id="appeal_out" class="output">Waiting...</pre>
              </div>
            </section>

            <section id="review_tab" class="panel hidden">
              <h2 class="section-title">Reviewer Tools</h2>
              <div id="review_access" class="access">Switch role to <b>Reviewer</b> to access queue and review operations.</div>
              <div id="review_content" class="hidden">
                <div class="grid">
                  <div class="card">
                    <button class="btn" id="queue_btn">Refresh Pending Appeals</button>
                    <div id="queue_list" class="review-list">Load queue to see pending appeals.</div>
                  </div>
                  <div class="card">
                    <div class="row"><label>Selected Appeal ID</label><input id="review_appeal_id" placeholder="select from queue or paste" /></div>
                    <div class="row"><label>Decision</label><select id="review_decision"><option value="approve">approve</option><option value="deny" selected>deny</option></select></div>
                    <div class="row"><label>Reviewer ID</label><input id="reviewer_id" value="human-reviewer-1" /></div>
                    <button class="btn secondary" id="review_btn">Submit Decision</button>
                    <pre id="review_out" class="output">Waiting...</pre>
                  </div>
                </div>
              </div>
            </section>

            <section id="verify_tab" class="panel hidden">
              <h2 class="section-title">Human Verification</h2>
              <div class="grid">
                <div class="card">
                  <div class="row"><label>Creator ID</label><input id="cert_creator" value="demo-author" /></div>
                  <button class="btn secondary" id="cert_load_submissions_btn">Load Creator Submissions</button>
                  <div class="row"><label>Select Submission</label><select id="cert_submission_id"><option value="">Load submissions first</option></select></div>
                  <div id="cert_submission_preview" class="hint">Choose a submission to generate a challenge tied to real content.</div>
                  <div class="row"><label>Paragraph/Section Hint</label><input id="cert_hint" value="opening argument paragraph" /></div>
                  <button class="btn" id="challenge_btn">1) Generate Challenge</button>
                  <div class="hint">Process-based challenge to verify authorship understanding.</div>
                  <pre id="challenge_out" class="output">Waiting...</pre>
                </div>
                <div class="card">
                  <div class="row"><label>Challenge Token</label><input id="challenge_token" placeholder="auto-filled from step 1" /></div>
                  <div class="row"><label>Attestation (required)</label><textarea id="cert_attest">I certify this submission is my original work.</textarea></div>
                  <div class="row"><label>Challenge Response</label><textarea id="challenge_response">I revised this section to connect each claim to evidence and removed weaker examples.</textarea></div>
                  <div class="row"><label>Active Writing Minutes</label><input id="e_active" type="number" value="48" /></div>
                  <div class="row"><label>Edit Sessions</label><input id="e_sessions" type="number" value="92" /></div>
                  <div class="row"><label>Major Revisions</label><input id="e_revisions" type="number" value="6" /></div>
                  <div class="row"><label>Max Paste Chars</label><input id="e_paste" type="number" value="80" /></div>
                  <div class="row"><label>Revision Notes Count</label><input id="e_notes" type="number" value="5" /></div>
                  <button class="btn secondary" id="cert_btn">2) Verify Certificate</button>
                  <pre id="cert_out" class="output">Waiting...</pre>
                </div>
              </div>
            </section>

            <section id="my_submissions_tab" class="panel hidden">
            <div class="section-title">My Submissions</div>

            <label>Creator ID</label>
            <input id="my_submission_creator" placeholder="creator id">

            <button class="btn secondary" id="my_submission_load">Load My Submissions</button>

            <div id="my_submission_tiles" class="tile-grid" style="margin-top:12px;">
                <div class="muted">No submissions loaded yet.</div>
            </div>
            </section>            
                            
            <section id="log_tab" class="panel hidden">
              <h2 class="section-title">Audit Log</h2>
              <div class="card">
                <div class="row"><label>Limit</label><input id="log_limit" value="10" /></div>
                <button class="btn" id="log_btn">Load Log Entries</button>
                <pre id="log_out" class="output">Waiting...</pre>
              </div>
            </section>

            <section id="analytics_tab" class="panel hidden">
              <h2 class="section-title">Analytics</h2>
              <div class="card">
                <p class="hint">Reviewer-facing analytics are on a separate page.</p>
                <p><a href="/dashboard" target="_blank">Open /dashboard</a></p>
                <p><a href="/analytics" target="_blank">Open raw /analytics JSON</a></p>
              </div>
            </section>
          </main>
        </div>
      </div>

      <script>
        const tabIds = ['submission_tab', 'appeal_tab', 'review_tab', 'verify_tab', 'my_submissions_tab', 'log_tab', 'analytics_tab'];

        function print(id, data) {
          const target = document.getElementById(id);
          target.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
        }

        function scoreClass(v) {
          if (typeof v !== 'number') return '';
          if (v >= 0.8) return 'score-high';
          if (v >= 0.5) return 'score-mid';
          return 'score-low';
        }

        function showTab(tabId) {
          tabIds.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.toggle('hidden', id !== tabId);
          });
          document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
          });
        }

        function renderSubmissionPretty(result) {
          const root = document.getElementById('submit_pretty');
          if (!root) return;
          if (!result || !result.body) {
            root.textContent = 'No result available yet.';
            return;
          }

          // Intercept rate limits or any other backend errors
          if (result.status === 429 || result.body.error) {
            root.innerHTML = `
              <div style="font-size:1.04rem;font-weight:700;color:#991b1b;">Submission Failed</div>
              <div style="margin-top:10px; padding:12px; background:#fee2e2; border:1px solid #fca5a5; border-radius:10px; color:#991b1b; font-weight: 500;">
                ⚠️ ${result.body.error || "You've been throttled. Please try again later."}
              </div>
            `;
            return; // Stop rendering the rest of the UI
          }

          const data = result.body;
          const cert = data.provenance_certificate || {};
          const score = typeof data.confidence === 'number' ? data.confidence.toFixed(3) : data.confidence;
          const llm = typeof data.llm_score === 'number' ? data.llm_score.toFixed(3) : data.llm_score;
          const stylo = typeof data.stylo_score === 'number' ? data.stylo_score.toFixed(3) : data.stylo_score;
         const structure = typeof data.structure_score === 'number' ? data.structure_score.toFixed(3) : (data.structure_score ?? 'n/a');

          let badgeHtml = '';
          if (cert.verified && cert.badge_image_url) {
            badgeHtml = `
              <div class="badge-wrap">
                <img src="${cert.badge_image_url}" alt="Human Verified badge" />
                <div>
                  <div style="font-weight:700;">Human Verified Certificate</div>
                  <div style="color:#4f676c;font-size:.85rem;">Certificate ID: ${cert.certificate_id || 'n/a'}</div>
                </div>
              </div>
            `;
          }

          root.innerHTML = `
            <div style="font-size:1.04rem;font-weight:700;">Submission Result (${result.status})</div>
            <div class="pretty-grid">
              <div class="metric"><div class="k">Attribution</div><div class="v ${scoreClass(data.confidence)}">${data.attribution || 'n/a'}</div></div>
              <div class="metric"><div class="k">Confidence</div><div class="v ${scoreClass(data.confidence)}">${score}</div></div>
              <div class="metric"><div class="k">LLM</div><div class="v ${scoreClass(data.llm_score)}">${llm}</div></div>
              <div class="metric"><div class="k">Stylometric</div><div class="v ${scoreClass(data.stylo_score)}">${stylo}</div></div>
              <div class="metric"><div class="k">Structure</div><div class="v ${scoreClass(data.structure_score)}">${structure}</div></div>
            </div>
            <div class="label-box">${data.label || 'No label returned.'}</div>
            ${badgeHtml}
          `;
        }

        function applyRole(role) {
          const reviewerOnly = role === 'reviewer';
          const access = document.getElementById('review_access');
          const content = document.getElementById('review_content');
          if (access && content) {
            access.classList.toggle('hidden', reviewerOnly);
            content.classList.toggle('hidden', !reviewerOnly);
          }
          document.querySelectorAll('.reviewer-only').forEach(el => el.classList.toggle('hidden', !reviewerOnly));
          document.querySelectorAll('.submitter-only').forEach(el => el.classList.toggle('hidden', reviewerOnly));
          const activeBtn = document.querySelector('.nav-btn.active');
          if (activeBtn && activeBtn.classList.contains('hidden')) {
            showTab(reviewerOnly ? 'review_tab' : 'submission_tab');
          }
        }

        async function requestJson(path, method, body) {
        const options = {
            method,
            headers: { "Content-Type": "application/json" },
        };
        if (body) options.body = JSON.stringify(body);

        function printResult(elId, result) {
        const target = document.getElementById(elId);
        if (!target) return;

        if (result.status === 429) {
            target.classList.add("rate-limit");
        } else {
            target.classList.remove("rate-limit");
        }

        const msg =
            result.status === 429
            ? `Rate limited (429): ${result.body?.error || "Too many requests"}`
            : JSON.stringify(result.body, null, 2);

        target.textContent = `HTTP ${result.status}\n${msg}`;
        }

        const res = await fetch(path, options);
        let payload;
        try {
            payload = await res.json();
        } catch {
            payload = { raw: await res.text() };
        }

        return {
            status: res.status,
            ok: res.ok,
            body: payload,
        };
        }

        document.querySelectorAll('.nav-btn').forEach(btn => btn.addEventListener('click', () => showTab(btn.dataset.tab)));
        document.getElementById('role_select').addEventListener('change', (e) => applyRole(e.target.value));

        document.getElementById('submission_mode').addEventListener('change', (e) => {
          const multimodal = e.target.value === 'image_metadata';
          document.getElementById('text_inputs').classList.toggle('hidden', multimodal);
          document.getElementById('multimodal_inputs').classList.toggle('hidden', !multimodal);
        });

        document.getElementById('submit_btn').onclick = async () => {
          const mode = document.getElementById('submission_mode').value;
          const creatorId = document.getElementById('submit_creator').value;
          let result;
          if (mode === 'text') {
            result = await requestJson('/submit', 'POST', {
              creator_id: creatorId,
              text: document.getElementById('submit_text').value,
            });
          } else {
            const tags = document.getElementById('mm_tags').value.split(',').map(s => s.trim()).filter(Boolean);
            result = await requestJson('/submit_multimodal', 'POST', {
              creator_id: creatorId,
              content_type: 'image_metadata',
              title: document.getElementById('mm_title').value,
              description: document.getElementById('mm_desc').value,
              tags,
            });
          }
          renderSubmissionPretty(result);
          print('submit_out', result);
          if (result.body && result.body.content_id) {
            document.getElementById('appeal_content_id').value = result.body.content_id;
          }
        };

        document.getElementById('submit_json_toggle').onclick = () => {
          const out = document.getElementById('submit_out');
          const btn = document.getElementById('submit_json_toggle');
          const hidden = out.classList.toggle('hidden');
          btn.textContent = hidden ? 'Show Full JSON' : 'Hide Full JSON';
        };

        function renderMySubmissions(rows) {
        const root = document.getElementById("my_submission_tiles");
        if (!rows || !rows.length) {
            root.innerHTML = `<div class="muted">No submissions found for this creator.</div>`;
            return;
        }

        root.innerHTML = rows.map(r => {
            const statusClass =
            r.final_state === "flagged"
                ? "status-ai"
                : r.final_state === "repealed"
                ? "status-repealed"
                : r.final_state === "in_review"
                ? "status-uncertain"
                : r.attribution === "uncertain"
                ? "status-uncertain"
                : "status-human";

            const statusText =
            r.final_state === "flagged"
                ? "AI Flagged"
                : r.final_state === "repealed"
                ? "Appeal Overturned"
                : r.final_state === "in_review"
                ? "Appeal in Review"
                : r.attribution === "uncertain"
                ? "Uncertain"
                : "Human";

            return `
            <div class="tile-card">

            <div class="tile-header">
                <div class="tile-id">
                    Submission ${r.content_id}
                    <svg class="copy-icon" onclick="navigator.clipboard.writeText('${r.content_id}'); this.style.color='#15803d'; setTimeout(() => this.style.color='', 1000);" title="Copy ID" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                </div>
                <div class="status-pill ${statusClass}">
                    ${statusText}
                </div>
            </div>

            <div class="preview">
            ${r.preview || ""}
            </div>

            <div class="tile-metrics">

            <div class="metric-box">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">
            ${Number(r.confidence).toFixed(3)}
            </div>
            </div>

            <div class="metric-box">
            <div class="metric-label">Appeal</div>
            <div class="metric-value">
            ${r.appeal_status}
            </div>
            </div>

            <div class="metric-box">
            <div class="metric-label">Label</div>
            <div class="metric-value">
            ${r.attribution}
            </div>
            </div>

            <div class="metric-box">
            <div class="metric-label">Submitted</div>
            <div class="metric-value">
            ${new Date(r.timestamp).toLocaleDateString()}
            </div>
            </div>

            </div>

            <div class="verify-row">

            <div>
            <b>Human Verification</b><br>
            ${r.human_verified ? "Verified" : "Not Verified"}
            </div>

            ${
            r.human_verified
            ?
            `<img src="/assets/human_verif.png">`
            :
            ""
            }

            </div>

            </div>
            `;
        }).join("");
        }

        document.getElementById("my_submission_load").onclick = async () => {
        const creatorId = document.getElementById("my_submission_creator").value.trim();
        const root = document.getElementById("my_submission_tiles");
        if (!creatorId) {
            root.innerHTML = `<div class="muted">Enter a creator ID first.</div>`;
            return;
        }

        const res = await fetch(`/creator_submissions?creator_id=${encodeURIComponent(creatorId)}`);
        const data = await res.json();
        renderMySubmissions(data.submissions || []);
        };

        document.getElementById('appeal_load_btn').onclick = async () => {
          const creatorId = document.getElementById('appeal_creator').value;
          if (!creatorId) return;
          const res = await fetch('/creator_submissions?creator_id=' + encodeURIComponent(creatorId));
          const body = await res.json();
          const select = document.getElementById('appeal_content_id');
          const flagged = (body.submissions || []).filter(r => r.final_state === 'flagged');
          
          if (!flagged.length) {
            select.innerHTML = '<option value="">No flagged submissions found.</option>';
            return;
          }
          select.innerHTML = flagged.map(r => `<option value="${r.content_id}">${r.content_id} | Score: ${Number(r.confidence).toFixed(3)}</option>`).join('');
        };

        document.getElementById('appeal_btn').onclick = async () => {
          const result = await requestJson('/appeal', 'POST', {
            content_id: document.getElementById('appeal_content_id').value,
            creator_id: document.getElementById('appeal_creator').value,
            creator_reasoning: document.getElementById('appeal_reason').value,
          });
          print('appeal_out', result);
          if (result.body && result.body.appeal_id) {
            document.getElementById('review_appeal_id').value = result.body.appeal_id;
          }
        };

        document.getElementById('queue_btn').onclick = async () => {
          const result = await requestJson('/appeals', 'GET');
          print('review_out', result);
          const list = document.getElementById('queue_list');
          const appeals = (result.body && result.body.appeals) ? result.body.appeals : [];
          if (!appeals.length) {
            list.textContent = 'No pending appeals.';
            return;
          }
          list.innerHTML = appeals.map(a => `
            <div class="review-item" data-appeal-id="${a.appeal_id}">
              <div><b>${a.appeal_id}</b></div>
              <div style="font-size:.82rem;color:#587a6d;">${a.content_id}</div>
              <div style="font-size:.82rem;color:#587a6d;">creator: ${a.creator_id} | score: ${a.confidence}</div>
            </div>
          `).join('');
          list.querySelectorAll('.review-item').forEach(item => {
            item.addEventListener('click', () => {
              document.getElementById('review_appeal_id').value = item.dataset.appealId;
            });
          });
        };

        document.getElementById('review_btn').onclick = async () => {
          const appealId = document.getElementById('review_appeal_id').value;
          const result = await requestJson('/appeals/' + encodeURIComponent(appealId) + '/review', 'POST', {
            decision: document.getElementById('review_decision').value,
            reviewer_id: document.getElementById('reviewer_id').value,
          });
          print('review_out', result);
          if (result.status === 200) {
            document.getElementById('queue_btn').click();
          }
        };

        document.getElementById('cert_load_submissions_btn').onclick = async () => {
          const creatorId = document.getElementById('cert_creator').value;
          const res = await fetch('/creator_submissions?creator_id=' + encodeURIComponent(creatorId));
          const body = await res.json();
          const select = document.getElementById('cert_submission_id');
          const preview = document.getElementById('cert_submission_preview');
          const rows = body.submissions || [];
          if (!rows.length) {
            select.innerHTML = '<option value="">No submissions found for creator</option>';
            preview.textContent = 'No submissions available for this creator id.';
            return;
          }
          select.innerHTML = rows.map(row => `<option value="${row.content_id}">${row.content_id} | ${row.attribution} | ${(Number(row.confidence) || 0).toFixed(3)}</option>`).join('');
          const first = rows[0];
          preview.textContent = first.preview || 'No preview text available.';
          select.onchange = () => {
            const chosen = rows.find(r => r.content_id === select.value);
            preview.textContent = chosen ? (chosen.preview || 'No preview text available.') : 'No preview text available.';
          };
        };

        document.getElementById('challenge_btn').onclick = async () => {
          const selectedContentId = document.getElementById('cert_submission_id').value;
          if (!selectedContentId) {
            print('challenge_out', { status: 400, body: { error: 'Please load and select a submission first.' } });
            return;
          }
          const result = await requestJson('/certificate/challenge', 'POST', {
            creator_id: document.getElementById('cert_creator').value,
            content_id: selectedContentId,
            paragraph_hint: document.getElementById('cert_hint').value,
          });
          print('challenge_out', result);
          if (result.body && result.body.challenge_token) {
            document.getElementById('challenge_token').value = result.body.challenge_token;
          }
        };

        document.getElementById('cert_btn').onclick = async () => {
          const result = await requestJson('/certificate/verify', 'POST', {
            creator_id: document.getElementById('cert_creator').value,
            attestation: document.getElementById('cert_attest').value,
            challenge_token: document.getElementById('challenge_token').value,
            challenge_response: document.getElementById('challenge_response').value,
            evidence: {
              active_minutes: Number(document.getElementById('e_active').value || 0),
              edit_sessions: Number(document.getElementById('e_sessions').value || 0),
              major_revisions: Number(document.getElementById('e_revisions').value || 0),
              max_paste_chars: Number(document.getElementById('e_paste').value || 0),
              revision_notes_count: Number(document.getElementById('e_notes').value || 0),
            },
          });
          print('cert_out', result);
        };

        document.getElementById('log_btn').onclick = async () => {
          const limit = Number(document.getElementById('log_limit').value || 10);
          const res = await fetch('/log?limit=' + encodeURIComponent(limit));
          const payload = await res.json();
          print('log_out', { status: res.status, body: payload });
        };

        applyRole('submitter');
        showTab('submission_tab');
      </script>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    app.run(port=5000, debug=True)
