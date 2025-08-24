import json


def normalize_query_payload(payload) -> str:
    """Normalize various Action Input shapes into a plain query string.

    Accepts:
    - plain string ("search terms")
    - JSON string ("{\"q\": \"...\"}")
    - dict-like object with keys 'input', 'q', 'query', or 'question'
    - other types (falls back to str(payload))
    """
    parsed = None
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except Exception:
            parsed = None

    elif isinstance(payload, dict):
        parsed = payload

    query = None
    if isinstance(parsed, dict):
        for k in ("input", "q", "query", "question"):
            if k in parsed and parsed[k] is not None:
                query = parsed[k]
                break
        if query is None and len(parsed) > 0:
            try:
                first_key = next(iter(parsed))
                query = parsed[first_key]
            except Exception:
                query = str(parsed)

    if query is None:
        query = str(payload)

    if not isinstance(query, str):
        query = str(query)

    return query


if __name__ == '__main__':
    samples = [
        "Mercedes Sosa studio albums 2000-2009",
        '{"q": "Mercedes Sosa studio albums 2000-2009"}',
        '{"input": "Mercedes Sosa studio albums 2000-2009"}',
        {"q": "Mercedes Sosa studio albums 2000-2009"},
        {"query": "Mercedes Sosa studio albums 2000-2009"},
        {"unexpected": "Mercedes Sosa studio albums 2000-2009"},
        12345,
    ]

    for s in samples:
        print("INPUT:", repr(s))
        print("NORMALIZED:", normalize_query_payload(s))
        print("---")
