# Git Surgeon

**Git Surgeon** is an AI-assisted command-line tool that allows developers to rewrite and clean up Git commit history using natural language.

With a simple prompt, you can squash, reorder, reword, or remove commits — all without manually invoking complex Git commands. `git-surgeon` brings intuitive, human-readable Git history editing to your workflow.

---

## ✨ Features

- 🔍 **Natural Language Interface**  
  Describe what you want in plain English — no more memorizing Git plumbing commands.

- 🩺 **Smart Commit Surgery**  
  Automatically squashes, drops, reorders, or rewrites commits according to your prompt.

- 🧠 **Powered by AI**  
  Uses large language models to interpret your instructions and generate precise Git operations.

- 🛠️ **Safe and Transparent**  
  Displays the exact Git commands before executing, so you stay in control.

---

## 📦 Installation

> Coming soon: Installation via `pip` or `brew`.  
> For now, clone the repo and run locally:

```bash
git clone https://github.com/yourusername/git-surgeon.git
cd git-surgeon
pip install -r requirements.txt
python surgeon.py
````

---

## 🚀 Usage

```bash
$ git-surgeon "Squash the last 4 commits into one and reword the message to 'Refactor and cleanup'"
```

The tool will:

1. Parse your instruction with AI
2. Show the proposed changes (e.g., interactive rebase)
3. Ask for confirmation before applying

---

## 🧪 Example Use Cases

* `"Drop the third commit from the top"`
* `"Reword the last commit to be more descriptive"`
* `"Reorder the last 5 commits by timestamp"`
* `"Split the second-to-last commit into two"`

---

## 🧠 How It Works

`git-surgeon` leverages a local Git history parser combined with an LLM backend (e.g., OpenAI GPT or local LLMs) to translate natural language into concrete Git commands. These are then executed in a controlled, previewable way.

---

## ⚠️ Disclaimer

This tool modifies Git history. Always ensure your branches are backed up or pushed before making destructive changes.

---

## 📄 License

MIT License © James Tan

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve functionality or language parsing.
