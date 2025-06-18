"""
Intent Training Data Generator for git-surgeon

This module generates synthetic training data for the git-surgeon NLP parser.
It creates varied, realistic command examples for each supported Git operation intent.

Features:
- Generates unique examples with realistic parameter substitution
- Ensures type-safe replacements to prevent semantically invalid examples
- Provides balanced coverage across all intent types
- Supports test/preview mode for rapid development

Usage:
    # Generate the full dataset
    python -m surgeon.parser.generator
    
    # Generate a test set for preview
    python -m surgeon.parser.generator --test
    
    # Run directly 
    python surgeon/parser/generator.py
"""

import sys
import os
from random import choice, randint, sample, shuffle
import csv
import itertools
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Optional, Any, Union, Callable

# Add the project root directory to the Python path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from surgeon modules
from surgeon.intents import Intent, get_required_params

# Generate at least 1000 unique samples 
SAMPLES_PER_INTENT = 200  # This will give us at least 1800 samples total

# Type definitions for parameter values
ParamTypes = {
    "n": "number",
    "branch": "branch_name",
    "oldbranch": "branch_name",
    "message": "commit_message",
    "file": "file_name",
    "hash": "commit_hash",
    "hash2": "commit_hash"
}

# Common parameter generators
def generate_branch_names() -> List[str]:
    """Generate a list of realistic Git branch names."""
    prefixes = ["feature/", "bugfix/", "hotfix/", "release/", "", ""]
    names = ["user-auth", "payment", "dashboard", "login", "cart", "checkout", "profile", 
             "search", "api", "notification", "settings", "admin", "main", "master", "dev", 
             "develop", "staging", "production", "test", "integration"]
    return [f"{prefix}{name}" for prefix in prefixes for name in names]

def generate_commit_messages() -> List[str]:
    """Generate a list of realistic Git commit messages."""
    actions = ["Add", "Fix", "Update", "Refactor", "Remove", "Implement", "Optimize", "Improve"]
    components = ["login form", "payment gateway", "user interface", "database schema", 
                  "authentication system", "API endpoints", "error handling", "performance", 
                  "security vulnerability", "documentation", "unit tests"]
    details = ["", "with better error handling", "to fix #123", "for better performance", 
               "to support new features", "according to design specifications", ""]
    
    messages = []
    for action, component in itertools.product(actions, components):
        for detail in details:
            if detail:
                messages.append(f"{action} {component} {detail}")
            else:
                messages.append(f"{action} {component}")
    return messages

def generate_file_names() -> List[str]:
    """Generate a list of realistic file paths."""
    directories = ["src/", "app/", "lib/", "utils/", "components/", "models/", "services/", ""]
    file_types = ["controller", "service", "model", "view", "utils", "helper", "component", "test", "config"]
    extensions = [".js", ".ts", ".py", ".rb", ".go", ".java", ".cpp", ".css", ".html"]
    
    return [f"{directory}{file_type}{extension}" for directory in directories 
            for file_type in file_types for extension in extensions]

# Safe parameter replacement function
def safe_replace(template: str, placeholder: str, value: str, param_type: str) -> str:
    """
    Safely replace a placeholder in a template with a value of the expected type.
    
    Args:
        template: The string template containing placeholders
        placeholder: The placeholder key (without braces) to replace
        value: The value to insert
        param_type: The expected parameter type
        
    Returns:
        The template with the placeholder replaced by the value
    """
    # Type validation based on parameter type
    if param_type == "number":
        if not str(value).isdigit():
            raise ValueError(f"Expected number for {placeholder}, got {value}")
    elif param_type == "commit_hash":
        # Simple validation for hash-like string
        if not all(c in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid hash format for {placeholder}: {value}")
    
    # Replace the placeholder with the value
    return template.replace(f"{{{placeholder}}}", str(value))

TEMPLATES = {
    Intent.SQUASH: [
        "squash the last {n} commits",
        "squash last {n} commits",
        "combine last {n} commits",
        "combine the last {n} commits",
        "merge previous {n} commits into one",
        "collapse {n} commits into one",
        "squash {n} commits together",
        "join last {n} commits",
        "make one commit from the last {n}",
        "consolidate {n} recent commits",
        "squash commits {n} through HEAD",
        "combine {n} commits with message '{message}'",
        "squash the last {n} commits with message '{message}'",
        "roll up last {n} commits into one"
    ],
    Intent.REWORD: [
        "reword the last commit",
        "change the message of the last commit",
        "edit the commit message",
        "update the commit message",
        "reword commit message to '{message}'",
        "change commit message to '{message}'",
        "fix the last commit message",
        "update commit message with '{message}'",
        "edit message of the most recent commit",
        "revise the previous commit's message",
        "modify the last commit's description",
        "correct typo in the last commit message",
        "rewrite the last commit's message"
    ],
    Intent.DROP: [
        "drop the last commit",
        "remove the previous commit",
        "delete the latest commit",
        "discard the most recent commit",
        "eliminate the last commit",
        "get rid of the previous commit",
        "remove commit {hash}",
        "drop commit from yesterday",
        "delete unnecessary commit",
        "remove broken commit",
        "discard commit with message '{message}'",
        "abandon the last pushed commit"
    ],
    Intent.REORDER: [
        "reorder the last {n} commits",
        "swap the last 2 commits",
        "change the order of recent commits",
        "move commit {hash} before {hash2}",
        "put the third commit first",
        "reverse the order of the last {n} commits",
        "reorder commits to prioritize bug fixes",
        "change commit sequence",
        "rearrange the last few commits",
        "move older commits to the top",
        "switch positions of commits {hash} and {hash2}"
    ],
    Intent.SPLIT: [
        "split the last commit",
        "break up the recent commit",
        "divide the last commit into parts",
        "separate the previous commit",
        "split commit {hash} into multiple commits",
        "break the last commit into logical chunks",
        "divide last commit into separate changes",
        "split up large commit",
        "make multiple commits from the last one",
        "break down monolithic commit"
    ],
    Intent.AMEND: [
        "amend the last commit",
        "fix the last commit",
        "add changes to the previous commit",
        "include {file} in the last commit",
        "update the last commit with new changes",
        "add forgotten file to previous commit",
        "amend commit with new files",
        "add {file} to the last commit",
        "change the last commit to include new changes",
        "fix typo in last commit",
        "amend last commit with message '{message}'",
        "modify the previous commit to add {file}"
    ],
    Intent.UNDO: [
        "undo the last commit",
        "revert the previous change",
        "rollback the last commit",
        "undo commit {hash}",
        "restore to before last commit",
        "cancel the most recent commit",
        "revert changes from yesterday",
        "undo the merge commit",
        "take back last commit but keep changes",
        "reset to before the last commit",
        "undo the commit that broke the build",
        "revert commit with message '{message}'"
    ],
    Intent.RENAME_BRANCH: [
        "rename the branch to {branch}",
        "change branch name to {branch}",
        "rename current branch to {branch}",
        "change the name of this branch to {branch}",
        "rename branch from {oldbranch} to {branch}",
        "update branch name to {branch}",
        "call this branch {branch} instead",
        "set branch name to {branch}",
        "change current branch's name to {branch}",
        "modify branch name from {oldbranch} to {branch}"
    ],
    Intent.REBASE_ONTO: [
        "rebase onto {branch}",
        "move commits onto {branch}",
        "rebase changes onto {branch}",
        "rebase current branch onto {branch}",
        "rebase {oldbranch} onto {branch}",
        "replay commits on top of {branch}",
        "apply commits from here onto {branch}",
        "transplant changes onto {branch}",
        "rebase last {n} commits onto {branch}",
        "move our changes on top of {branch}",
        "rebase changes since {hash} onto {branch}"
    ]
}

def generate_data(filename: str = "data/intents.tsv", limit: Optional[int] = None, 
                 dry_run: bool = False) -> Dict[str, Any]:
    """
    Generate training data for intent classification.
    
    Args:
        filename: Path where the output TSV file should be written
        limit: Optional limit on samples per intent (for testing)
        dry_run: If True, print samples instead of writing to file
        
    Returns:
        Statistics about the generated dataset
    """
    if not dry_run:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Generate parameter values
    branch_names = generate_branch_names()
    commit_messages = generate_commit_messages()
    file_names = generate_file_names()
    commit_hashes = [f"{i:07x}" for i in range(1, 1000)]
    
    # Track used examples to ensure uniqueness
    used_examples = set()
    # Track statistics for reporting
    stats = {
        "total": 0,
        "by_intent": defaultdict(int),
        "unique_templates_used": defaultdict(set),
        "attempts": 0
    }
    
    samples_needed = limit or SAMPLES_PER_INTENT
    
    # Prepare for writing or previewing
    if dry_run:
        preview_samples = defaultdict(list)
    else:
        f = open(filename, "w", newline="")
        writer = csv.writer(f, delimiter="\t")
    
    try:
        for intent, templates in TEMPLATES.items():
            count = 0
            attempts = 0
            
            while count < samples_needed and attempts < samples_needed * 5:
                attempts += 1
                stats["attempts"] += 1
                template = choice(templates)
                example = template
                
                # Track which templates were used
                stats["unique_templates_used"][intent.name].add(template)
                
                # Replace placeholders with appropriate values
                try:
                    if "{n}" in template:
                        example = safe_replace(example, "n", str(randint(2, 10)), ParamTypes["n"])
                    
                    if "{branch}" in template:
                        example = safe_replace(example, "branch", choice(branch_names), ParamTypes["branch"])
                    
                    if "{oldbranch}" in template:
                        # Make sure oldbranch != branch to avoid "rename branch from X to X"
                        old_branch = choice(branch_names)
                        if "{branch}" in template:
                            while old_branch in example:
                                old_branch = choice(branch_names)
                        example = safe_replace(example, "oldbranch", old_branch, ParamTypes["oldbranch"])
                    
                    if "{message}" in template:
                        example = safe_replace(example, "message", choice(commit_messages), ParamTypes["message"])
                    
                    if "{file}" in template:
                        example = safe_replace(example, "file", choice(file_names), ParamTypes["file"])
                    
                    if "{hash}" in template:
                        example = safe_replace(example, "hash", choice(commit_hashes), ParamTypes["hash"])
                    
                    if "{hash2}" in template:
                        # Make sure hash2 is different from hash
                        hash2 = choice(commit_hashes)
                        while hash2 in example:
                            hash2 = choice(commit_hashes)
                        example = safe_replace(example, "hash2", hash2, ParamTypes["hash2"])
                
                except ValueError as e:
                    # Skip this template if parameter replacement failed
                    print(f"Warning: {e}")
                    continue
                
                # Only add if unique
                if example not in used_examples:
                    if dry_run:
                        preview_samples[intent.name].append(example)
                    else:
                        writer.writerow([example, intent.name])
                    
                    used_examples.add(example)
                    count += 1
                    stats["total"] += 1
                    stats["by_intent"][intent.name] += 1
        
        # Calculate variance metrics
        stats["template_coverage"] = {}
        for intent_name, templates_used in stats["unique_templates_used"].items():
            # Look up the original intent enum by name to get the templates list
            for intent_enum in Intent:
                if intent_enum.name == intent_name:
                    stats["template_coverage"][intent_name] = len(templates_used) / len(TEMPLATES[intent_enum])
                    break
        
        # In dry_run mode, print sample output
        if dry_run:
            for intent_name, samples in preview_samples.items():
                print(f"\n=== {intent_name} ({len(samples)} samples) ===")
                for i, sample in enumerate(samples[:5]):  # Print first 5 samples
                    print(f"  {i+1}. {sample}")
                if len(samples) > 5:
                    print(f"  ...and {len(samples) - 5} more")
            
            print("\n=== Coverage Statistics ===")
            for intent_name, coverage in stats["template_coverage"].items():
                print(f"  {intent_name}: {coverage:.1%} template coverage")
        
        print(f"\n✅ Generated {stats['total']} unique examples" + 
              (f" in {filename}" if not dry_run else " (preview mode)"))
        
        # Report any intents with low variance
        low_variance_threshold = 0.7
        low_variance_intents = [
            intent for intent, coverage in stats["template_coverage"].items() 
            if coverage < low_variance_threshold
        ]
        
        if low_variance_intents:
            print("\n⚠️ Low template coverage for intents:")
            for intent in low_variance_intents:
                print(f"  - {intent}: {stats['template_coverage'][intent]:.1%} (Consider adding more templates)")
    
    finally:
        if not dry_run and 'f' in locals():
            f.close()
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for intent classification")
    parser.add_argument("--test", action="store_true", help="Run in test mode (prints samples without saving)")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per intent (for testing)")
    parser.add_argument("--output", type=str, default="data/intents.tsv", help="Output file path")
    
    args = parser.parse_args()
    
    if args.test:
        generate_data(limit=args.limit or 10, dry_run=True)
    else:
        generate_data(filename=args.output, limit=args.limit)
