# intent catalog + schemas

from enum import Enum

class Intent(Enum):
    SQUASH = "squash"
    REWORD = "reword"
    DROP = "drop"
    REORDER = "reorder"
    SPLIT = "split"
    AMEND = "amend"
    UNDO = "undo"
    RENAME_BRANCH = "rename-branch"
    REBASE_ONTO = "rebase-onto"

INTENT_SCHEMAS = {
    Intent.SQUASH: {
        "params": ["count", "new_message"]
    },
    Intent.REWORD: {
        "params": ["target", "new_message"]
    },
    Intent.DROP: {
        "params": ["target"]
    },
    Intent.REORDER: {
        "params": ["order"]
    },
    Intent.SPLIT: {
        "params": ["target", "split_strategy"]
    },
    Intent.AMEND: {
        "params": ["add_files", "new_message"]
    },
    Intent.UNDO: {
        "params": ["target"]
    },
    Intent.RENAME_BRANCH: {
        "params": ["new_name"]
    },
    Intent.REBASE_ONTO: {
        "params": ["base_branch", "commit_range"]
    }
}

def get_required_params(intent: Intent):
    return INTENT_SCHEMAS.get(intent, {}).get("params", [])

# E.X.: get_required_params(Intent.SQUASH)



