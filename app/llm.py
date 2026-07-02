"""
LLM orchestration for the assistant.

The pipeline is split into three pieces:

1. Route the message to one of four paths: order tool, product tool,
   knowledge-base answer, or direct chat.
2. Execute the retrieval / tool lookup needed for that path.
3. Use the configured free Gemini model or local Ollama model to turn the
   grounded result into the final response.

If neither provider is available, structured extractive fallbacks are used
instead of dumping raw document chunks.
"""
import json
import logging
import re
import time
from typing import Optional

from app.config import settings, validate_gemini_key
from app.tools import (
    get_order_status,
    search_product,
    text_mentions_catalog,
    extract_catalog_mentions,
    extract_last_catalog_mention,
    extract_last_catalog_user_message,
    detect_catalog_categories,
)
from app.retrieval import retrieve_for_knowledge, format_context, build_retrieval_query
from app.ingestion import get_collection

logger = logging.getLogger(__name__)

_last_llm_error: Optional[str] = None

_ORDER_ID_PATTERN = re.compile(r"\bORD\d+\b", re.IGNORECASE)
_CHEAPER_KEYWORDS = ["cheaper", "cheap", "less expensive", "lower price", "budget option"]
_PRODUCT_INTENT_PHRASES = (
    "show me",
    "looking for",
    "i need",
    "find",
    "do you have",
    "can i get",
    "any",
)
_CATALOG_HINTS = ("price", "prices", "stock", "catalog", "in stock", "out of stock")
_NAME_STATEMENT_PATTERN = re.compile(
    r"\bmy name is\s+(?P<name>[A-Za-z][A-Za-z\-' ]{0,60})",
    re.IGNORECASE,
)
_NAME_REVERSE_PATTERN = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z\-' ]{0,60})\s+is\s+my\s+name\b",
    re.IGNORECASE,
)
_NAME_CALL_ME_PATTERN = re.compile(
    r"\b(?:call me|i'm|i am|im)\s+(?P<name>[A-Za-z][A-Za-z\-' ]{0,60})",
    re.IGNORECASE,
)
_NAME_IT_IS_PATTERN = re.compile(
    r"\b(?:it's|it is|thats|that is|this is)\s+(?P<name>[A-Za-z][A-Za-z\-' ]{0,60})",
    re.IGNORECASE,
)
_QUESTION_STARTERS = (
    "what", "where", "when", "why", "how", "who", "which",
    "does", "is", "are", "can", "could", "would", "do you", "did",
)
_GREETINGS = ("hello", "hi ", "hi!", "hi,", "hey", "thanks", "thank you")
_DOCUMENT_HINTS = (
    "document",
    "uploaded",
    "file",
    "pdf",
    "md",
    "markdown",
    "according to",
    "in the doc",
    "in the document",
    "knowledge base",
)

_DOCUMENT_INTENT_PHRASES = (
    "tell me about",
    "tell about",
    "tell me something",
    "something about",
    "what does",
    "explain",
    "summarize",
    "summary of",
    "privacy",
    "policy",
    "loyalty",
    "company history",
    "support hours",
    "shipping",
    "return policy",
    "warranty",
    "what about",
    "how about",
)

_NAME_FOLLOW_UP_WORDS = {
    "remember",
    "please",
    "thanks",
    "thank",
    "okay",
    "ok",
    "right",
    "man",
    "buddy",
}

_INVALID_NAME_WORDS = {
    "what",
    "who",
    "where",
    "when",
    "why",
    "how",
    "which",
    "do",
    "does",
    "is",
    "are",
    "am",
    "can",
    "could",
    "would",
    "should",
    "your",
    "my",
    "name",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "but",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "tell",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "about",
    "something",
}

BASE_SYSTEM_PROMPT = """You are a helpful AI assistant.

Use the conversation history to remember user-specific facts and the
current discussion. Be concise, truthful, and do not invent details.
"""

KNOWLEDGE_SYSTEM_PROMPT = """You answer questions using only the uploaded document context.

Read the user's question carefully, including conversation history for context
(e.g. "cheaper options" may refer to a product discussed earlier).

Write a clear, natural answer in your own words. Focus only on what the user
asked for. Do not paste large blocks of raw document text.

If the context does not contain the answer, reply with exactly this sentence
and nothing else:
I couldn't find that information in the uploaded documents.
"""

TOOL_SYSTEM_PROMPT = """You summarize the output of a mock tool (orders or product catalog).

Important: the product catalog and order lookup are standalone systems. They are
NOT related to companies or topics from uploaded documents. Do not mention
uploaded files, or any knowledge-base content in your answer.

Use only the provided tool result JSON. Be concise and factual.
"""

FALLBACK_NOT_FOUND_MESSAGE = "I couldn't find that information in the uploaded documents."


def _looks_like_question(text: str) -> bool:
    """Heuristic: does this message look like it's asking something?"""
    t = text.strip().lower()
    return t.endswith("?") or t.startswith(_QUESTION_STARTERS)


def _looks_like_product_request(text: str, history: list[dict]) -> bool:
    """Detect catalog intent using live product data, not hardcoded SKUs."""
    lower = text.lower().strip()
    if any(kw in lower for kw in _CHEAPER_KEYWORDS):
        return (
            text_mentions_catalog(lower)
            or extract_last_catalog_user_message(history, exclude_latest=text) is not None
        )
    if text_mentions_catalog(lower):
        return True
    if any(hint in lower for hint in _CATALOG_HINTS) and text_mentions_catalog(lower):
        return True
    if any(phrase in lower for phrase in _PRODUCT_INTENT_PHRASES):
        return (
            text_mentions_catalog(lower)
            or extract_last_catalog_user_message(history, exclude_latest=text) is not None
        )
    return False


def _extract_last_product_keyword(history: list[dict]) -> Optional[str]:
    """Scan backwards through history for the last catalog term mentioned."""
    return extract_last_catalog_mention(history)


def _find_users_name(history: list[dict]) -> Optional[str]:
    for msg in history:
        if msg["role"] != "user":
            continue
        name = _extract_name(msg["content"])
        if name:
            return name
    return None


def _extract_name(text: str) -> Optional[str]:
    """Extract the user's name from a self-introduction and drop filler text.

    Examples:
    - "My name is Sadik" -> "Sadik"
    - "My name is Sadik remember it okay" -> "Sadik"
    - "Sadik is my name" -> "Sadik"
    - "It's Sadik Mahmud" -> "Sadik Mahmud"
    - "I'm John" -> "John"
    """
    if _looks_like_name_recall(text):
        return None

    match = (
        _NAME_STATEMENT_PATTERN.search(text)
        or _NAME_REVERSE_PATTERN.search(text)
        or _NAME_CALL_ME_PATTERN.search(text)
        or _NAME_IT_IS_PATTERN.search(text)
    )
    if not match:
        return None

    candidate = match.group("name").strip()
    candidate = re.split(
        r"\b(?:remember(?:\s+it)?(?:\s+okay)?|please|thanks?|thank\s+you|ok(?:ay)?|right|buddy|man)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    candidate = re.split(r"[,.!?;:]", candidate, maxsplit=1)[0].strip()

    words = candidate.split()
    while words and words[-1].lower() in _NAME_FOLLOW_UP_WORDS:
        words.pop()

    cleaned = " ".join(words).strip()
    if not cleaned:
        return None

    first_word = cleaned.split()[0].lower()
    if first_word in _INVALID_NAME_WORDS:
        return None

    return cleaned or None


def _strip_name_intro(text: str, name: str) -> Optional[str]:
    """Strip a name-introduction prefix from a combined message.

    Examples:
      "I'm Sadik. Now tell me about X" -> "Now tell me about X"
      "It's Sadik, tell me about X" -> "tell me about X"
      "My name is Sadik and tell me about X" -> "and tell me about X"
    """
    if not text or not name:
        return None

    patterns = [
        rf"^(?:hi|hey|hello)[,!\s]+my name is\s+{re.escape(name)}[,.!?;:]?\s*",
        rf"^(?:hi|hey|hello)[,!\s]+(?:i'?m|i am|im)\s+{re.escape(name)}[,.!?;:]?\s*",
        rf"\bi'?m\s+{re.escape(name)}[,.!?;:]?\s*",
        rf"\bi am\s+{re.escape(name)}[,.!?;:]?\s*",
        rf"\bit'?s\s+{re.escape(name)}[,.!?;:]?\s*",
        rf"\bmy name is\s+{re.escape(name)}[,.!?;:]?\s*",
        rf"\b{re.escape(name)}\s+is my name[,.!?;:]?\s*",
        rf"\bcall me\s+{re.escape(name)}[,.!?;:]?\s*",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            remaining = text[m.end() :].strip()
            if remaining:
                remaining = re.sub(
                    r"^(now|so|then|and|but|also|please)[,\s]*",
                    "",
                    remaining,
                    flags=re.IGNORECASE,
                ).strip()
                return remaining
    return None


def _looks_like_name_recall(text: str) -> bool:
    """User is asking the assistant to recall their name from session memory."""
    text_lower = text.lower().strip().rstrip("?").strip()
    recall_phrases = (
        "what is my name",
        "what's my name",
        "whats my name",
        "who am i",
        "remember my name",
        "do you know my name",
        "do you remember my name",
        "what did i say my name",
        "tell me my name",
    )
    if any(phrase in text_lower for phrase in recall_phrases):
        return True
    if "my name" in text_lower and _looks_like_question(text_lower):
        return _extract_name(text) is None
    return False


def _had_recent_document_discussion(history: list[dict]) -> bool:
    """True when the session was recently discussing uploaded document topics."""
    for msg in reversed(history[-8:]):
        content = str(msg.get("content", "")).lower()
        if msg.get("role") == "user" and _looks_like_document_request(content):
            return True
        if msg.get("role") == "assistant" and any(
            hint in content
            for hint in ("khadok", "policy", "warranty", "loyalty", "shipping", "return")
        ):
            return True
    return False


def _looks_like_document_question(text: str) -> bool:
    text_lower = text.lower()
    return any(hint in text_lower for hint in _DOCUMENT_HINTS + _DOCUMENT_INTENT_PHRASES)


def _looks_like_document_request(text: str) -> bool:
    if _looks_like_name_recall(text):
        return False

    text_lower = text.lower().strip()
    if any(hint in text_lower for hint in _DOCUMENT_HINTS + _DOCUMENT_INTENT_PHRASES):
        return True
    if text_lower.startswith(("tell me", "tell", "explain", "summarize")):
        return True
    doc_topics = (
        "company", "policy", "program", "warranty", "shipping", "return",
        "loyalty", "support", "hours", "khadok", "document", "refund",
    )
    if re.search(r"\b(tell|about|explain|describe)\b", text_lower):
        if any(topic in text_lower for topic in doc_topics):
            return get_collection().count() > 0
    if re.search(r"\bwhat\b", text_lower) and any(topic in text_lower for topic in doc_topics):
        return get_collection().count() > 0
    return False


def _looks_like_knowledge_follow_up(text: str, history: list[dict]) -> bool:
    text_lower = text.lower().strip()
    if any(text_lower.startswith(p) for p in ("what about", "how about")):
        return True
    if _is_short_document_follow_up(text_lower) and _had_recent_document_discussion(history):
        return True
    return False


def _is_short_document_follow_up(text_lower: str) -> bool:
    follow_terms = ("loyalty", "program", "policy", "warranty", "shipping", "return", "support", "hours")
    return any(term in text_lower for term in follow_terms) and len(_meaningful_query_terms(text_lower)) <= 5


def _history_to_text(history: list[dict]) -> str:
    lines = []
    for message in history:
        role = message.get("role", "user").capitalize()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no prior conversation)"


def _build_prompt(system_prompt: str, history: list[dict], latest_message: str, extra_block: str = "") -> str:
    sections = [
        system_prompt.strip(),
        "Conversation history:\n" + _history_to_text(history),
        "Current user message:\n" + latest_message.strip(),
    ]
    if extra_block.strip():
        sections.append(extra_block.strip())
    sections.append("Answer:")
    return "\n\n".join(sections)


def _resolve_product_search_query(history: list[dict], latest_message: str) -> str:
    """Build the catalog search string; follow-ups like 'cheaper options' reuse prior context."""
    lower = latest_message.lower().strip()
    if any(kw in lower for kw in _CHEAPER_KEYWORDS):
        prior = extract_last_catalog_user_message(history, exclude_latest=latest_message)
        if prior:
            return prior
        mention = extract_last_catalog_mention(history, exclude_latest=latest_message)
        if mention:
            cats = detect_catalog_categories(mention)
            if cats:
                return cats[0]
            return mention
    return latest_message


def _product_category_label(
    history: list[dict], latest_message: str, tool_result: dict
) -> str:
    """Human-readable category for product replies (never the full user sentence)."""
    groups = tool_result.get("groups") or []
    if len(groups) == 1:
        return groups[0]["category"]

    cats = detect_catalog_categories(latest_message)
    if len(cats) == 1:
        return cats[0]

    if any(kw in latest_message.lower() for kw in _CHEAPER_KEYWORDS):
        prior = extract_last_catalog_user_message(history, exclude_latest=latest_message)
        if prior:
            prior_cats = detect_catalog_categories(prior)
            if prior_cats:
                return prior_cats[-1]

    mentions = extract_catalog_mentions(latest_message)
    if mentions:
        for mention in mentions:
            cats = detect_catalog_categories(mention)
            if len(cats) == 1:
                return cats[0]

    return "product"


def _infer_product_query(history: list[dict], latest_message: str) -> str:
    return _resolve_product_search_query(history, latest_message)


def _sort_product_results(results: list[dict]) -> list[dict]:
    return sorted(results, key=lambda item: item.get("price", 0))


def _format_product_category(label: str) -> str:
    label = label.strip().lower()
    if label.endswith("s"):
        return label[:-1]
    return label


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _meaningful_query_terms(text: str) -> list[str]:
    terms = []
    for token in _tokenize(text):
        if token not in _STOPWORDS and len(token) > 1:
            terms.append(token)
    return terms


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _score_sentence(sentence: str, query_terms: list[str]) -> int:
    sentence_terms = set(_meaningful_query_terms(sentence))
    return sum(1 for term in query_terms if term in sentence_terms)


def _strip_section_header(text: str) -> str:
    text = text.strip()
    if text.startswith("#"):
        text = re.sub(r"^#{1,3}\s+[\w &]+", "", text, count=1).strip()
    return text


def _rank_chunks_for_query(query: str, chunks: list[dict]) -> list[dict]:
    """Rank chunks so section headers (e.g. ## Loyalty Program) win for topic queries."""
    query_terms = _meaningful_query_terms(query)
    if not query_terms:
        return chunks

    query_set = set(query_terms)
    scored: list[tuple[float, dict]] = []
    for chunk in chunks:
        text = chunk["text"]
        text_lower = text.lower()
        score = float(chunk.get("score", 0))

        if text.startswith("##"):
            header_terms = set(_meaningful_query_terms(text.split("\n", 1)[0]))
            overlap = query_set & header_terms
            if overlap:
                score += 10 * len(overlap)
            if query_set <= header_terms or all(t in text_lower for t in query_terms):
                score += 15

        score += sum(1.0 for term in query_terms if term in text_lower)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored]


def _extractive_answer(query: str, chunks: list[dict]) -> Optional[str]:
    """Pick the best-matching sentences from retrieved chunks (LLM-offline fallback)."""
    query_terms = _meaningful_query_terms(query)
    if not query_terms or not chunks:
        return None

    ranked = _rank_chunks_for_query(query, chunks)
    focus_chunks = ranked[:1]

    scored: list[tuple[int, str]] = []
    for chunk in focus_chunks:
        for sentence in _split_sentences(chunk["text"]):
            score = _score_sentence(sentence, query_terms)
            if score > 0:
                scored.append((score, sentence))

    if not scored:
        best = _strip_section_header(focus_chunks[0]["text"])
        return best[:500] if best else None

    scored.sort(key=lambda item: item[0], reverse=True)
    top_score = scored[0][0]
    best = [s for score, s in scored if score >= max(1, top_score - 1)]
    seen: set[str] = set()
    unique: list[str] = []
    for sentence in best[:6]:
        cleaned = _strip_section_header(sentence)
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        unique.append(cleaned)

    if unique:
        return " ".join(unique)

    body = _strip_section_header(focus_chunks[0]["text"])
    return body[:600] if body else None


def _fallback_knowledge_reply(query: str, chunks: list[dict]) -> str:
    """Structured answer when the LLM is unavailable — never dump raw chunks."""
    if not chunks:
        return FALLBACK_NOT_FOUND_MESSAGE

    extractive = _extractive_answer(query, chunks)
    if extractive:
        return extractive

    best = chunks[0]["text"].strip()
    sentences = _split_sentences(best)
    if sentences:
        return sentences[0]

    return best[:400] + ("..." if len(best) > 400 else "")


def _classify_llm_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "503" in msg or "unavailable" in msg:
        return "rate_limit"
    if "429" in msg or "quota" in msg or "rate" in msg or "resource exhausted" in msg:
        return "rate_limit"
    if "401" in msg or "403" in msg or "api key" in msg or "permission" in msg:
        return "auth_error"
    if "404" in msg or "not found" in msg:
        return "model_not_found"
    return "provider_error"


def _call_gemini(prompt: str) -> Optional[str]:
    global _last_llm_error
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        _last_llm_error = "Gemini API key is not configured."
        return None

    key_hint = validate_gemini_key(api_key)
    if key_hint:
        _last_llm_error = key_hint
        logger.error(_last_llm_error)
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=settings.LLM_TEMPERATURE,
                    ),
                )
                text = (getattr(response, "text", None) or "").strip()
                if text:
                    _last_llm_error = None
                    return text
                _last_llm_error = "Gemini returned an empty response."
                return None
            except Exception as exc:
                last_exc = exc
                error_kind = _classify_llm_error(exc)
                _last_llm_error = f"Gemini {error_kind}: {exc}"
                logger.warning("Gemini attempt %s failed: %s", attempt + 1, _last_llm_error)
                if error_kind == "rate_limit" and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break

        if last_exc:
            logger.error("Gemini unavailable after retries: %s", last_exc)
        return None
    except ImportError:
        _last_llm_error = "google-genai package not installed. Run: pip install google-genai"
        logger.error(_last_llm_error)
        return None
    except Exception as exc:
        _last_llm_error = f"Gemini setup error: {exc}"
        logger.error(_last_llm_error)
        return None


def _call_ollama(prompt: str) -> Optional[str]:
    global _last_llm_error
    try:
        import ollama

        client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        response = client.chat(
            model=settings.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": settings.LLM_TEMPERATURE},
        )
        message = response.get("message", {}) if isinstance(response, dict) else {}
        text = str(message.get("content", "")).strip()
        if text:
            _last_llm_error = None
            return text
        _last_llm_error = "Ollama returned an empty response."
        return None
    except Exception as exc:
        _last_llm_error = f"Ollama error: {exc}"
        logger.warning(_last_llm_error)
        return None


def _should_skip_ollama_fallback() -> bool:
    """Don't mask Gemini config errors with unrelated Ollama failures."""
    if not _last_llm_error:
        return False
    err = _last_llm_error.lower()
    return any(
        token in err
        for token in (
            "auth_error",
            "api key",
            "api_key_invalid",
            "not configured",
            "format looks invalid",
            "model_not_found",
            "google-genai package not installed",
        )
    )


def _generate_text(prompt: str) -> Optional[str]:
    requested = settings.LLM_PROVIDER

    if requested == "gemini":
        reply = _call_gemini(prompt)
        if reply:
            return reply
        if _should_skip_ollama_fallback():
            return None
        return _call_ollama(prompt)

    if requested == "ollama":
        reply = _call_ollama(prompt)
        if reply:
            return reply
        return _call_gemini(prompt)

    return None


def get_runtime_status() -> dict:
    """Expose the configured and effective LLM provider for /health."""
    configured = settings.LLM_PROVIDER
    api_key = settings.GEMINI_API_KEY
    key_hint = validate_gemini_key(api_key) if api_key else "GEMINI_API_KEY is not set in .env"

    if configured == "gemini" and api_key and not key_hint:
        effective = "gemini"
        model = settings.GEMINI_MODEL
    elif configured == "ollama":
        effective = "ollama"
        model = settings.OLLAMA_MODEL
    elif api_key and not key_hint:
        effective = "gemini"
        model = settings.GEMINI_MODEL
    else:
        effective = "ollama"
        model = settings.OLLAMA_MODEL

    return {
        "configured_provider": configured,
        "effective_provider": effective,
        "model": model,
        "gemini_configured": bool(api_key),
        "gemini_key_hint": key_hint,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "last_error": _last_llm_error,
    }


def get_last_llm_error() -> Optional[str]:
    return _last_llm_error


def _format_order_reply(tool_result: dict) -> str:
    if tool_result.get("found"):
        return (
            f"Order {tool_result['order_id']} is currently '{tool_result['status']}'. "
            f"Estimated delivery: {tool_result.get('estimated_delivery') or 'N/A'}."
        )
    return tool_result.get("message", "No matching order was found.")


def _product_result_lines(results: list[dict]) -> list[str]:
    return [
        f"- {item['name']}: ${item['price']} ({'in stock' if item['in_stock'] else 'out of stock'})"
        for item in _sort_product_results(results)
    ]


def _format_product_search_reply(
    tool_result: dict, history: list[dict], latest_message: str
) -> str:
    """Format catalog results without blending in RAG / document context."""
    if not tool_result.get("found"):
        return tool_result.get(
            "message", "No matching products were found in the catalog."
        )

    if any(kw in latest_message.lower() for kw in _CHEAPER_KEYWORDS):
        results = _sort_product_results(tool_result.get("results", []))
        category = _product_category_label(history, latest_message, tool_result)
        cheapest = results[:2]
        cheap_lines = _product_result_lines(cheapest)
        return f"Here are some cheaper {category} options:\n" + "\n".join(cheap_lines)

    groups = tool_result.get("groups") or []
    if len(groups) > 1:
        sections = []
        for group in groups:
            cat = group["category"]
            lines = _product_result_lines(group.get("results", []))
            if lines:
                sections.append(f"Here are some {cat} options:\n" + "\n".join(lines))
        if sections:
            return "\n\n".join(sections)

    results = _sort_product_results(tool_result.get("results", []))
    lines = _product_result_lines(results)
    category = _product_category_label(history, latest_message, tool_result)
    if category != "product":
        return f"Here are some {category} options:\n" + "\n".join(lines)

    return "Here are the options I found:\n" + "\n".join(lines)


def _tool_reply(tool_name: str, tool_result: dict, history: list[dict], latest_message: str) -> str:
    # Product catalog is independent of uploaded documents — format directly
    # so conversation context (e.g. Khadok Store) never leaks into listings.
    if tool_name == "search_product":
        return _format_product_search_reply(tool_result, history, latest_message)

    if tool_name == "get_order_status":
        deterministic = _format_order_reply(tool_result)
        prompt = _build_prompt(
            TOOL_SYSTEM_PROMPT,
            [],
            latest_message,
            f"Tool name: {tool_name}\nTool result:\n{json.dumps(tool_result, indent=2)}",
        )
        reply = _generate_text(prompt)
        return reply or deterministic

    tool_block = json.dumps(tool_result, indent=2)
    prompt = _build_prompt(
        TOOL_SYSTEM_PROMPT,
        [],
        latest_message,
        f"Tool name: {tool_name}\nTool result:\n{tool_block}",
    )

    reply = _generate_text(prompt)
    if reply:
        return reply

    return json.dumps(tool_result)


def _knowledge_reply(
    history: list[dict],
    latest_message: str,
    retrieved_chunks: list[dict],
) -> tuple[str, bool]:
    """
    Answer from the knowledge base. Returns (reply, llm_degraded).
    llm_degraded is True when the LLM was unavailable and a structured fallback was used.
    """
    chunks = list(retrieved_chunks)
    if not chunks:
        chunks = retrieve_for_knowledge(latest_message, history=history)

    if not chunks:
        return FALLBACK_NOT_FOUND_MESSAGE, False

    context = format_context(chunks)
    prompt = _build_prompt(
        KNOWLEDGE_SYSTEM_PROMPT,
        history,
        latest_message,
        f"Uploaded document context:\n{context}",
    )

    reply = _generate_text(prompt)
    if reply:
        if reply.strip() == FALLBACK_NOT_FOUND_MESSAGE and chunks:
            extractive = _fallback_knowledge_reply(latest_message, chunks)
            if extractive != FALLBACK_NOT_FOUND_MESSAGE:
                return extractive, True
        return reply, False

    logger.warning(
        "LLM unavailable for knowledge reply; using extractive fallback. Last error: %s",
        _last_llm_error,
    )
    return _fallback_knowledge_reply(latest_message, chunks), True


def _direct_reply(history: list[dict], latest_message: str) -> tuple[str, bool]:
    prompt = _build_prompt(BASE_SYSTEM_PROMPT, history, latest_message)
    reply = _generate_text(prompt)
    if reply:
        return reply, False

    if _looks_like_question(latest_message.lower()) or any(
        p in latest_message.lower() for p in _PRODUCT_INTENT_PHRASES
    ):
        return FALLBACK_NOT_FOUND_MESSAGE, bool(_last_llm_error)

    return "I can help with uploaded documents, order status, and product search.", bool(_last_llm_error)


def _route_request(history: list[dict], latest_message: str) -> tuple[str, Optional[dict]]:
    text = latest_message.lower().strip()

    order_match = _ORDER_ID_PATTERN.search(latest_message)
    if order_match:
        return "tool_order", {"order_id": order_match.group(0)}

    if _looks_like_name_recall(latest_message):
        return "name_recall", None

    name_statement = (
        _NAME_STATEMENT_PATTERN.search(latest_message)
        or _NAME_REVERSE_PATTERN.search(latest_message)
        or _NAME_CALL_ME_PATTERN.search(latest_message)
        or _NAME_IT_IS_PATTERN.search(latest_message)
    )
    if name_statement:
        extracted_name = _extract_name(latest_message)
        if extracted_name:
            remaining = _strip_name_intro(latest_message, extracted_name)
            if remaining:
                return "name_then_continue", {"name": extracted_name, "remaining": remaining}
            return "name_statement", {"name": extracted_name}

    if _looks_like_product_request(latest_message, history):
        if not _looks_like_document_request(latest_message):
            return "tool_product", None

    if _looks_like_document_request(latest_message) or _looks_like_knowledge_follow_up(text, history):
        return "knowledge", None

    if text.startswith(_GREETINGS) and len(text.split()) <= 4:
        return "greeting", None

    if _had_recent_document_discussion(history) and _is_short_document_follow_up(text):
        return "knowledge", None

    return "direct", None


# ---------------------------------------------------------------------------
# Public entry point used by main.py
# ---------------------------------------------------------------------------
def generate_response(history: list[dict], latest_message: str) -> dict:
    """
    Route first (no embedding), then retrieve only for knowledge paths.
    """
    route, payload = _route_request(history, latest_message)
    llm_degraded = False
    retrieved_chunks: list[dict] = []
    used_retrieval = False
    used_tool = None

    if route == "tool_order":
        tool_result = get_order_status(payload["order_id"])
        reply = _tool_reply("get_order_status", tool_result, history, latest_message)
        used_tool = "get_order_status"
    elif route == "tool_product":
        query = _resolve_product_search_query(history, latest_message)
        tool_result = search_product(query)
        reply = _tool_reply("search_product", tool_result, history, latest_message)
        used_tool = "search_product"
    elif route == "name_statement":
        reply = f"Nice to meet you, {payload['name']}!"
    elif route == "name_then_continue":
        name = payload["name"]
        remaining = payload["remaining"]
        greeting = f"Nice to meet you, {name}!"
        route2, payload2 = _route_request(history, remaining)
        if route2 == "knowledge":
            retrieved_chunks = retrieve_for_knowledge(remaining, history=history)
            used_retrieval = len(retrieved_chunks) > 0
            reply2, degraded = _knowledge_reply(history, remaining, retrieved_chunks)
            llm_degraded = llm_degraded or degraded
        elif route2 == "tool_product":
            query = _resolve_product_search_query(history, remaining)
            tool_result = search_product(query)
            reply2 = _tool_reply("search_product", tool_result, history, remaining)
        elif route2 == "tool_order":
            tool_result = get_order_status(payload2.get("order_id", ""))
            reply2 = _tool_reply("get_order_status", tool_result, history, remaining)
        elif route2 == "direct":
            reply2, degraded = _direct_reply(history, remaining)
            llm_degraded = llm_degraded or degraded
        else:
            reply2 = remaining
        reply = f"{greeting}\n\n{reply2}"
    elif route == "name_recall":
        name = _find_users_name(history)
        reply = f"Your name is {name}." if name else "You haven't told me your name yet."
    elif route == "greeting":
        reply = "Hello! How can I help you today?"
    elif route == "knowledge":
        retrieved_chunks = retrieve_for_knowledge(latest_message, history=history)
        used_retrieval = len(retrieved_chunks) > 0
        reply, degraded = _knowledge_reply(history, latest_message, retrieved_chunks)
        llm_degraded = llm_degraded or degraded
    else:
        reply, degraded = _direct_reply(history, latest_message)
        llm_degraded = llm_degraded or degraded

    return {
        "reply": reply,
        "used_tool": used_tool,
        "sources": retrieved_chunks if used_retrieval else [],
        "used_retrieval": used_retrieval,
        "llm_degraded": llm_degraded,
        "llm_error": _last_llm_error if llm_degraded else None,
        "route": route,
    }