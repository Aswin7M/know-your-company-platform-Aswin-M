# Ollama Setup Guide

This guide explains how to install and prepare Ollama for the Know Your Company project.

Know Your Company uses Ollama as the local LLM runtime. The project is configured to use `qwen3:1.7b` by default so that it can run on a Windows machine with limited RAM and without a paid LLM API.

## 1. Install Ollama

Download and install Ollama for Windows from the official Ollama website:

https://ollama.com/download/windows

Complete the Windows installation using the default options unless you have a specific reason to change them.

After installation, open a new Command Prompt or VS Code terminal.

## 2. Verify Ollama

Run:

```cmd
ollama --version
```

A version number should be displayed.

If Windows reports that `ollama` is not recognized, close and reopen the terminal. If it still fails, restart Windows and try again.

## 3. Download the Qwen3 Model

Know Your Company uses:

```text
qwen3:1.7b
```

Download it with:

```cmd
ollama pull qwen3:1.7b
```

The model is downloaded to Ollama's local model storage. It is not included in the GitHub repository.

## 4. Test the Model

Run:

```cmd
ollama run qwen3:1.7b
```

Then enter a simple test such as:

```text
Explain what a healthcare technology company does in one sentence.
```

If the model responds, the local LLM is working.

To exit the model:

```text
/bye
```

## 5. Verify the Ollama Service

Ollama normally exposes its local service at:

```text
http://localhost:11434
```

You can test it from Command Prompt with:

```cmd
curl http://localhost:11434/api/tags
```

A successful response should contain information about the locally available models.

If `curl` is unavailable in your environment, you can simply verify the service by running:

```cmd
ollama list
```

You should see `qwen3:1.7b` in the model list.

## 6. Configure Know Your Company

The project uses the following default Ollama configuration:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_TIMEOUT=300
OLLAMA_NUM_CTX=4096
```

The application can use these defaults without additional configuration.

If you need to override them, use the project's `.env` file.

For example:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
```

Do not commit `.env` files containing machine-specific or private configuration.

## 7. Start Know Your Company

After Ollama and the model are ready, open the project folder in VS Code and run:

```cmd
setup_windows.bat
```

Then start the application:

```cmd
run.bat
```

The Streamlit application will open locally in your browser.

Ollama should remain installed and available while the application is using the local model.

## 8. Important: What Is and Is Not Included in GitHub

The GitHub repository contains the application source code and documentation.

It does not contain:

- The Ollama Windows installation
- The `qwen3:1.7b` model files
- The Python `.venv` virtual environment
- Downloaded embedding model files
- Runtime company research data
- Local vector-store data

These are created or downloaded locally on each machine.

Therefore, someone cloning the repository on another computer must install the prerequisites before running the application.

## 9. Recommended Setup Order

For a fresh Windows machine, use this order:

```text
1. Install Python
2. Install Ollama
3. Verify Ollama
4. Download qwen3:1.7b
5. Clone/download Know Your Company
6. Open the project in VS Code
7. Run setup_windows.bat
8. Run run.bat
```

Or, in command form:

```cmd
ollama --version
ollama pull qwen3:1.7b
ollama list
setup_windows.bat
run.bat
```

## 10. Troubleshooting

### `ollama` is not recognized

Close and reopen Command Prompt or VS Code after installing Ollama.

If the command is still unavailable, restart Windows and run:

```cmd
ollama --version
```

### The model is missing

Run:

```cmd
ollama list
```

If `qwen3:1.7b` is not listed, run:

```cmd
ollama pull qwen3:1.7b
```

### The application cannot connect to Ollama

Verify that Ollama is installed and that the local service is available.

Run:

```cmd
ollama list
```

Then verify:

```cmd
curl http://localhost:11434/api/tags
```

Also check that the project's `.env` configuration, if present, uses:

```env
OLLAMA_BASE_URL=http://localhost:11434
```

### Local inference is slow

`qwen3:1.7b` is intentionally used as a small local model for lower-resource machines. CPU-only inference can still take time, especially during larger research runs.

Avoid switching to a substantially larger model on an 8 GB RAM machine unless the system has enough available memory.

## 11. Local-First Design

Know Your Company is designed around a local-first workflow:

- The LLM runs through local Ollama.
- No OpenAI, Anthropic, Gemini, Groq, Tavily, or SerpAPI API key is required for the default workflow.
- Company research and application data are stored locally.
- Web access is used for public company research.
- The application is intended to run on the user's own Windows machine.

This setup keeps the initial operating cost at zero apart from the user's existing computer and internet connection.

## Reference

Official Ollama website:

https://ollama.com/

Official Ollama Windows download:

https://ollama.com/download/windows
