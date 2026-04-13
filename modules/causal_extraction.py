import json
import re


def extract_causal_pairs(text_chunks, api_key):
    """
    Extract cause-effect pairs from text chunks using GPT-3.5-turbo.
    Returns a list of dicts: [{"cause": ..., "effect": ..., "confidence": ...}]
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        combined_text = " ".join(text_chunks)

        if not combined_text.strip():
            return []

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a causal reasoning expert. Extract direct cause-effect relationships. "
                        "Return ONLY a JSON array. Each item MUST have 'cause', 'effect', and 'confidence' (0-1)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Extract causal pairs from this text. format: [{{'cause': '...', 'effect': '...', 'confidence': 0.9}}]\n"
                        f"Text: {combined_text}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        raw_content = response.choices[0].message.content.strip()

        # Robust JSON extraction: Find the first '[' and last ']'
        json_match = re.search(r"(\[.*\])", raw_content, re.DOTALL)
        if json_match:
            cleaned = json_match.group(1)
        else:
            cleaned = raw_content.replace("```json", "").replace("```", "").strip()

        # Parse JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try replacing single quotes with double quotes if LLM used them incorrectly
            try:
                fixed_quotes = cleaned.replace("'", '"')
                parsed = json.loads(fixed_quotes)
            except:
                return _manual_extract(text_chunks)

        # Validate and filter
        valid_pairs = []
        if isinstance(parsed, list):
            for item in parsed:
                if (
                    isinstance(item, dict)
                    and "cause" in item
                    and "effect" in item
                    and str(item.get("cause", "")).strip() != ""
                    and str(item.get("effect", "")).strip() != ""
                ):
                    valid_pairs.append({
                        "cause": str(item["cause"]).strip(),
                        "effect": str(item["effect"]).strip(),
                        "confidence": float(item.get("confidence", 0.8)) 
                    })
        
        if not valid_pairs:
            return _manual_extract(text_chunks)

        return valid_pairs

    except Exception as e:
        error_msg = str(e).lower()
        if "invalid api key" in error_msg or "authentication" in error_msg:
            raise ValueError("Invalid API key. Please check your key.")
        elif "rate limit" in error_msg:
            raise ValueError("API rate limit reached. Please wait 60 seconds.")
        else:
            print(f"[Causal Extraction] Error: {e}")
            return _manual_extract(text_chunks)


def _manual_extract(text_chunks):
    """
    Fallback: improved manual keyword handler.
    """
    causal_keywords = [
        ("caused by", False),
        ("causes", True),
        ("caused", True),
        ("leads to", True),
        ("led to", True),
        ("resulting in", True),
        ("results in", True),
        ("resulted in", True),
        ("triggered", True),
        ("because of", False),
        ("due to", False),
    ]

    pairs = []
    combined = " ".join(text_chunks)
    # Split into sentences or clauses
    segments = re.split(r'[.\n;]', combined)

    for segment in segments:
        seg_lower = segment.lower()
        for keyword, cause_first in causal_keywords:
            if keyword in seg_lower:
                idx = seg_lower.find(keyword)
                if cause_first:
                    cause = segment[:idx].strip().strip(",").strip()
                    effect = segment[idx + len(keyword):].strip().strip(",").strip()
                else:
                    effect = segment[:idx].strip().strip(",").strip()
                    cause = segment[idx + len(keyword):].strip().strip(",").strip()

                if cause and effect and len(cause) > 3 and len(effect) > 3:
                    pairs.append({"cause": cause, "effect": effect, "confidence": 0.6})
                break

    return pairs


def format_causal_chain(causal_pairs):
    """
    Format the extracted causal pairs into a readable causal chain string.
    Returns a string like: "Root Cause → Effect1 → Effect2 → Final Effect"
    """
    if not causal_pairs:
        return "No causal chain found."

    try:
        # Sort by confidence descending
        sorted_pairs = sorted(causal_pairs, key=lambda x: x.get("confidence", 0), reverse=True)

        # Collect all causes and effects
        all_causes = {p["cause"] for p in causal_pairs}
        all_effects = {p["effect"] for p in causal_pairs}

        # Root causes: things that appear as cause but never as effect
        root_causes = all_causes - all_effects

        # Build adjacency map
        cause_to_effect = {}
        for p in causal_pairs:
            if p["cause"] not in cause_to_effect:
                cause_to_effect[p["cause"]] = p["effect"]

        # Traverse the chain from each root cause
        best_chain = []
        for root in root_causes:
            chain = [root]
            current = root
            visited = {root}
            while current in cause_to_effect:
                nxt = cause_to_effect[current]
                if nxt in visited:
                    break
                chain.append(nxt)
                visited.add(nxt)
                current = nxt
            if len(chain) > len(best_chain):
                best_chain = chain

        if best_chain:
            return " → ".join(best_chain)

        # Fallback: just list all causes and effects with arrows
        chain_parts = []
        for p in sorted_pairs[:5]:
            chain_parts.append(f"{p['cause']} → {p['effect']}")
        return " | ".join(chain_parts)

    except Exception as e:
        print(f"[Chain Format] Error: {e}")
        # Ultra fallback
        return " → ".join(
            [f"{p['cause']} → {p['effect']}" for p in causal_pairs[:3]]
        )
