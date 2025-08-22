# CrewAI SEC Data Pipeline

A Python-based pipeline that uses [CrewAI](https://github.com/joaomdmoura/crewAI), [LiteLLM](https://docs.litellm.ai/), and other tools to:
- Fetch recent SEC filings (last 48 hours) via EDGAR Atom feeds
- Aggregate data and perform lightweight sentiment analysis using a chosen LLM (Groq, Hugging Face, etc.)
- Save a consolidated report (JSON and CSV)

---

## Features

- **Modular agents** for data fetching, processing, and reporting
- **Configurable LLM provider and model** via CLI (`--model`) or environment variable

---

## Requirements

- Python **3.10+**
- A [Groq](https://console.groq.com/) or [Hugging Face](https://huggingface.co/) API key (for sentiment analysis)
- Basic familiarity with Python and virtual environments

---

## Installation

1. **Clone this repo:**
   ```bash
   git clone https://github.com/siddharthk08/CrowdWisdomTrading-AI-Agent.git
   cd CrowdWisdomTrading-AI-Agent

2. **Create and activate a virtual environment:**

- python -m venv venv
- source venv/bin/activate   # On Windows: venv\Scripts\activate


3.**Install dependencies:**

- pip install --upgrade pip
- pip uninstall numpy -y
- pip install "numpy<1.28" "scipy<1.12" scikit-learn
- pip install feedparser requests python-dotenv litellm crewai youtube-transcript-api beautifulsoup4 pandas

#Configuration

1.**Environment Variables**

2.**Create a .env file (or export vars) with:**

- GROQ_API_KEY=your_api_key_here          # or HUGGINGFACE_API_KEY if using huggingface provider
- GROQ_MODEL=groq/llama-3.1-8b-instant   # or another provider/model
- SEC_USER_AGENT=you@example.com CrewAI-SEC-Task

#Command-line Arguments
```bash
 python main.py

