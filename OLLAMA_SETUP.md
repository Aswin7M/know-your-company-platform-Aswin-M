# Ollama setup for Know Your Company

Know Your Company uses [Ollama](https://ollama.com) to run a language model **locally on your own computer**.

* **Free.** Ollama and the models are free to download and run.
* **No API key, no account, no credit card.**
* **Private.** Your research stays on your machine; nothing is sent to a cloud LLM.
* **Speed depends on your hardware.** On a CPU-only laptop a small model answers slowly (seconds to minutes per answer).
* **8 GB RAM means small models.** Larger models (7B and up) may be very slow or fail to load. Start small.

## 1. Download Ollama

Go to <https://ollama.com/download> and download the **Windows** installer.

## 2. Install Ollama

Run the installer. Ollama starts automatically and runs in the background (look for the llama icon in the system tray). It listens on `http://localhost:11434`.

## 3. Verify the installation

Open **Command Prompt** or **PowerShell**:

```bash
ollama --version
```

You should see a version number.

## 4. Pull a small model

The recommended starting model for an 8 GB RAM / CPU-only machine is **qwen3:1.7b**:

```bash
ollama pull qwen3:1.7b
```

The download is roughly 1-2 GB. Other small options you can try later (check <https://ollama.com/library> for what is currently available, as the catalogue changes):

| Model | Notes |
|---|---|
| `qwen3:1.7b` | Recommended default. Small and instruction-following. |
| `llama3.2:1b` / `llama3.2:3b` | Alternatives; 3b is slower and uses more RAM. |
| `gemma3:1b` | Another very small option. |
| `qwen3:4b` | Better quality but noticeably heavier - may be too slow on 8 GB. |

Model names are configurable - nothing is hard-coded. Set `OLLAMA_MODEL` in `.env` or change it on the **Settings** page.

## 5. Test the model

```bash
ollama run qwen3:1.7b
```

Type a question, press Enter, and type `/bye` to exit. If you get an answer, the model works.

## 6. Start Know Your Company

1. Run `setup_windows.bat` once (installs Python dependencies).
2. Double-click `run.bat`.
3. Open **Settings** in the app. You should see **🟢 Ollama Connected**.

## Troubleshooting

| Symptom | Fix |
|---|---|
| **🔴 Ollama Not Available** | Start Ollama from the Start menu, or run `ollama serve` in a terminal. Check `OLLAMA_BASE_URL` in `.env`. |
| **🟡 Model missing** | Run `ollama pull <model>` for the model shown in the app. |
| Timeouts / very slow answers | Use a smaller model, close other apps, lower `TOP_K`, or raise `OLLAMA_TIMEOUT` in `.env`. |
| Out-of-memory or the model fails to load | Use a smaller model (`qwen3:1.7b`, `llama3.2:1b`). Lower `OLLAMA_NUM_CTX`. |
| Answers are vague or wrong | Small models are limited. The app cross-checks extracted facts against the evidence and drops unsupported ones, but chat answers are still only as good as the model. Always check the cited sources. |

If you are only exploring, you can tick off **Run AI analysis** when researching a company: the app will still collect, clean and index public evidence without calling the model.
