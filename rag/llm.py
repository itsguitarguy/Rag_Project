"""Step 5: the only API call. Strict answer-from-excerpts + code-level citation check."""
import json
import re

from .config import GEMINI_MODEL, LLM_PROVIDER, MAX_TOKENS, MODEL, NOT_FOUND
from .text_utils import normalize

SYSTEM_PROMPT = f"""You are a document question-answering assistant.
RULES:
1. Answer ONLY using the excerpts in <document_excerpts>. No outside knowledge, guesses or assumptions.
2. If the excerpts do not contain the answer, set "answer" to exactly "{NOT_FOUND}" and "citations" to [].
3. If only partly answered, give the supported part and state what is missing.
4. Every claim needs a citation: the excerpt number and a SHORT quote copied EXACTLY from that excerpt.
Respond with ONLY valid JSON: {{"answer": "...", "citations": [{{"excerpt": 1, "quote": "..."}}]}}"""


def verify_citations(citations, hits):
    """Keep only citations whose quote really appears in the cited chunk."""
    verified = []
    for c in citations:
        try:
            chunk = hits[int(c["excerpt"]) - 1]
        except (KeyError, ValueError, IndexError, TypeError):
            continue
        quote = c.get("quote", "")
        if quote and normalize(quote) in normalize(chunk["text"]):
            verified.append({"location": chunk["location"], "quote": quote})
    return verified


def _call_gemini(system, user):
    from google import genai            # pip install google-genai
    from google.genai import types
    client = genai.Client()             # reads GEMINI_API_KEY from environment
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json"))
    return resp.text


def _call_claude(system, user):
    import anthropic                    # reads ANTHROPIC_API_KEY from environment
    resp = anthropic.Anthropic().messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": user}])
    return resp.content[0].text


def ask_llm(question, hits):
    excerpts = "\n\n".join(
        f"[Excerpt {n} | {c['location']}]\n{c['text']}" for n, c in enumerate(hits, 1))
    user = f"<document_excerpts>\n{excerpts}\n</document_excerpts>\n\nQuestion: {question}"
    call = _call_gemini if LLM_PROVIDER == "gemini" else _call_claude
    text = call(SYSTEM_PROMPT, user)

    raw = re.sub(r"^```(?:json)?|```$", "", (text or "").strip()).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return "Could not produce a verifiable answer. Please rephrase."

    if data.get("answer", "").strip() == NOT_FOUND:
        return NOT_FOUND

    citations = verify_citations(data.get("citations", []), hits)
    if not citations:                      # answer with no verifiable evidence -> reject
        return NOT_FOUND

    sources = "\n".join(f'  - [{c["location"]}] "{c["quote"]}"' for c in citations)
    return f'{data["answer"]}\n\nSources:\n{sources}'
