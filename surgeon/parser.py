# Natural language → intent parser
import re
from rapidfuzz import process, fuzz
from surgeon.intents import Intent

INTENT_KEYWORDS = {
    Intent.SQUASH: ["squash", "combine commits", "merge commits"],
    Intent.REWORD: ["reword", "change message", "edit commit message"],
    Intent.DROP: ["drop", "remove commit", "delete commit"],
    Intent.REORDER: ["reorder", "change order", "move commits", "swap"],
    Intent.SPLIT: ["split", "break up commit"],
    Intent.AMEND: ["amend", "modify last commit", "edit last commit"],
    Intent.UNDO: ["undo", "revert last change"],
    Intent.RENAME_BRANCH: ["rename branch", "change branch name"],
    Intent.REBASE_ONTO: ["rebase onto", "move commits onto"]
}


def detect_intent(text: str):
    candidates = []
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            # Use partial_ratio for more flexible matching
            score = process.extractOne(text, [keyword], scorer=fuzz.partial_ratio)[1]
            candidates.append((intent, score))
    best_intent, best_score = max(candidates, key=lambda x: x[1])
    # Lower threshold to 60 for more permissive matching
    if best_score > 60:
        return best_intent
    else:
        return None


class IntentParseError(Exception):
    pass

def parse_intent(command: str):
    """Parse a natural language string into an intent and its parameters."""
    command_lower = command.strip().lower()

    intent = detect_intent(command_lower)
    if intent is None:
        raise IntentParseError("Could not understand the command confidently.")

    # Dispatch based on detected intent
    if intent == Intent.SQUASH:
        return _parse_squash(command_lower)
    elif intent == Intent.REWORD:
        return _parse_reword(command_lower)
    elif intent == Intent.DROP:
        return _parse_drop(command_lower)
    elif intent == Intent.REORDER:
        return _parse_reorder(command_lower)
    elif intent == Intent.SPLIT:
        return _parse_split(command_lower)
    elif intent == Intent.AMEND:
        return _parse_amend(command_lower)
    elif intent == Intent.UNDO:
        return _parse_undo(command_lower)
    elif intent == Intent.RENAME_BRANCH:
        return _parse_rename_branch(command_lower)
    elif intent == Intent.REBASE_ONTO:
        return _parse_rebase_onto(command_lower)

    raise IntentParseError("Intent recognized but no parser available.")


# Basic individual parsers

def _parse_squash(text):
    match = re.search(r"last\s+(\d+)\s+commits?", text)
    count = int(match.group(1)) if match else 2
    msg_match = re.search(r"to ['\"](.+?)['\"]", text)
    message = msg_match.group(1) if msg_match else "Squashed commits"
    return {
        "intent": Intent.SQUASH,
        "params": {
            "count": count,
            "new_message": message
        }
    }

def _parse_reword(text):
    msg_match = re.search(r"['\"](.+?)['\"]", text)
    message = msg_match.group(1) if msg_match else "Updated message"
    return {
        "intent": Intent.REWORD,
        "params": {
            "target": "HEAD",
            "new_message": message
        }
    }

def _parse_drop(text):
    return {
        "intent": Intent.DROP,
        "params": {
            "target": "HEAD~1"
        }
    }

def _parse_reorder(text):
    return {
        "intent": Intent.REORDER,
        "params": {
            "order": ["HEAD~2", "HEAD", "HEAD~1"]
        }
    }

def _parse_split(text):
    return {
        "intent": Intent.SPLIT,
        "params": {
            "target": "HEAD",
            "split_strategy": "interactive"
        }
    }

def _parse_amend(text):
    return {
        "intent": Intent.AMEND,
        "params": {
            "add_files": ["."],
            "new_message": "Amended commit"
        }
    }

def _parse_undo(text):
    return {
        "intent": Intent.UNDO,
        "params": {
            "target": "HEAD"
        }
    }

def _parse_rename_branch(text):
    match = re.search(r"to\s+([a-zA-Z0-9_\-/]+)", text)
    name = match.group(1) if match else "new-branch"
    return {
        "intent": Intent.RENAME_BRANCH,
        "params": {
            "new_name": name
        }
    }

def _parse_rebase_onto(text):
    match = re.search(r"onto\s+([a-zA-Z0-9_\-/]+)", text)
    branch = match.group(1) if match else "main"
    return {
        "intent": Intent.REBASE_ONTO,
        "params": {
            "base_branch": branch,
            "commit_range": "HEAD~5..HEAD"
        }
    }
