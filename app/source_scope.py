"""Conservative exclusions for documented, easily confused AWS feature scopes.

This curated rule set is not a universal semantic relevance classifier. Preserve
raw search results in the bundle; restrict only the evidence offered to generation.
"""

import re


def matches_task(title, english_search_phrase):
    title, query = title.lower(), english_search_phrase.lower()
    if "macie" in title and "macie" not in query:
        return False
    if "web function" in title and "web function" not in query:
        return False
    if "lambda function code" in title or "deployment package" in title:
        if not re.search(r"\b(deploy\w*|package|revision)\b|function code", query):
            return False
    return True
