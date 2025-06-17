# Natural language → intent parser
import re
from surgeon.intents import Intent

class IntentParseError(Exception):
    pass

def parse_intent(command: str):
    """Parse a natural language string into an intent and its parameters."""
    command = command.strip().lower()

    if "squash" in command:
        return _parse_squash(command)
    elif "reword" in command or "change message" in command:
        return _parse_reword(command)
    elif "drop" in command or "remove" in command:
        return _parse_drop(command)
    elif "reorder" in command or "move" in command or "swap" in command:
        return _parse_reorder(command)
    elif "split" in command:
        return _parse_split(command)
    elif "amend" in command:
        return _parse_amend(command)
    elif "undo" in command:
        return _parse_undo(command)
    elif "rename" in command and "branch" in command:
        return _parse_rename_branch(command)
    elif "rebase" in command and "onto" in command:
        return _parse_rebase_onto(command)
    
    raise IntentParseError("Could not understand the command.")

# 👇 Basic individual parsers

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
