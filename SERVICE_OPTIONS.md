# Catalan Lecture Processor — Service Deployment Options

How to offer this tool as a permanent service to IQS students, so they just
go to a URL, upload audio, and get results back.

---

## What the service needs to run

| Resource | Minimum | Recommended |
|---|---|---|
| **GPU** | 16 GB VRAM (T4) | 24 GB VRAM (A10/L4/RTX 3090) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 30 GB (models + OS) | 50 GB (room for cached audio/outputs) |
| **CPU** | 4 cores | 8 cores |
| **Network** | Public IP or reverse proxy | HTTPS with domain name |

### GPU memory breakdown (models loaded sequentially, not simultaneously)

| Step | Model | VRAM (fp16) |
|---|---|---|
| Transcription | faster-whisper-large-v3 | ~3.5 GB |
| Translation | NLLB-200 3.3B | ~6.6 GB |
| Summarization (if local LLM) | Llama 3.1 8B | ~5 GB |
| **Peak** (one model at a time) | | **~6.6 GB** |

Models are loaded/unloaded sequentially, so a single 16 GB GPU handles everything.
With 24 GB you could keep multiple models loaded and skip load/unload time.

### Self-hosted LLM for summarization (replaces Gemini)

On Colab we rely on the Gemini API for text cleanup and summarization. When
self-hosting, we can run an open-source LLM locally via **Ollama** instead —
the codebase already supports it as a backend (it checks Ollama first before
Gemini). This means **zero external API dependencies**.

Setup is simple:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.1:8b    # 4.7 GB download, fits any 12+ GB GPU

# That's it — the app auto-detects Ollama at localhost:11434
```

#### What fits on which GPU

Models load sequentially (Whisper → unload → NLLB → unload → LLM), so you
only need VRAM for the single largest model at any time.

| GPU | VRAM | Processing flow | Summarization quality |
|---|---|---|---|
| **RTX 3060** | 12 GB | Whisper (3.5 GB) → NLLB 3.3B (6.6 GB) → Llama 3.1 8B Q4 (5 GB) | Good |
| **RTX 3090** | 24 GB | Same, but can keep 2 models loaded simultaneously | Very good |
| **RTX 3090** | 24 GB | Can also run Llama 3.1 70B Q4 (fits in 24 GB) | Excellent |
| **2x RTX 3090** | 48 GB | Llama 3.1 70B fp16 | Best possible |

#### Recommended open-source LLM models

| Model | Size (fp16) | Size (Q4) | Multilingual | Notes |
|---|---|---|---|---|
| **Llama 3.1 8B** | ~16 GB | ~4.7 GB | Yes | Best balance of quality/speed, recommended default |
| **Mistral 7B** | ~14 GB | ~4.1 GB | Yes | Fast, good at structured output (summaries) |
| **Llama 3.1 70B** | ~140 GB | ~40 GB | Yes | Significantly better quality, needs 24+ GB GPU for Q4 |
| **BSC/AINA Catalan models** | varies | varies | Catalan-focused | Natural fit, worth asking BSC about availability |
| **Phi-3 Mini (3.8B)** | ~7.6 GB | ~2.2 GB | Limited | Smallest option, works on low VRAM, weaker multilingual |

**Recommendation:** Start with **Llama 3.1 8B** via Ollama. It handles
Catalan, Spanish, English, Portuguese and Italian well, fits on any 12+ GB
GPU, and produces good quality summaries. Upgrade to 70B if you have a
24 GB+ GPU and want the best quality.

#### Realistic quality expectations for local LLMs

Summarization is one of the easier tasks for LLMs — it's extracting key
points, not creative writing or complex reasoning. That said, model size
matters for certain scenarios.

| Scenario | Gemini Flash | Llama 3.1 8B | Llama 3.1 70B |
|---|---|---|---|
| English summaries | Excellent | Good | Excellent |
| Spanish summaries | Excellent | Good | Very good |
| Catalan summaries | Very good | Decent | Good |
| Technical vocabulary | Excellent | Sometimes misses nuance | Very good |
| Long lectures (map-reduce) | Excellent | Can lose context between chunks | Very good |

**Where 8B may struggle:**
- Very technical/domain-specific lectures (niche terminology in chemistry, law, etc.)
- Summarizing in Catalan specifically (smaller language = less training data)
- Long lectures where map-reduce chunking loses cross-section context

**Where 8B is fine:**
- Summarizing in English or Spanish (well represented in training data)
- General academic content (humanities, social sciences, introductory courses)
- Extracting key terms and bullet points

#### Recommended approach: hybrid (best of both worlds)

Rather than choosing between local LLM or cloud API, use both:

- **Transcription + translation** — always local (Whisper + NLLB, already works great)
- **Summarization** — prefer Gemini when available (free, better quality), fall back to local Ollama when offline, rate-limited, or if privacy is a concern

The codebase already supports this. Currently it checks Ollama first, then
Gemini. For the hybrid approach, flip the priority:

1. Gemini API (if key is set and internet is available)
2. Ollama (if running locally)
3. Skip summarization (if neither is available)

This gives you:
- **Best quality summaries** when internet is available (Gemini)
- **Full offline capability** when it's not (Ollama)
- **No single point of failure** — if Gemini goes down or gets rate-limited, Ollama takes over
- **Privacy option** — don't set the Gemini key if you want all data to stay local

#### Self-hosted vs Gemini API comparison

| | Self-hosted (Ollama 8B) | Self-hosted (Ollama 70B) | Gemini API |
|---|---|---|---|
| **Cost** | Free (your GPU) | Free (your GPU) | Free tier (rate limited) |
| **Privacy** | Data stays local | Data stays local | Data sent to Google |
| **Internet required** | No | No | Yes |
| **Summary quality** | Good | Excellent | Very good |
| **Catalan quality** | Decent | Good | Very good |
| **GPU needed** | 12 GB+ | 24 GB+ | None (cloud) |
| **Setup** | Install Ollama + pull model | Same | Get API key, set env var |
| **Rate limits** | None | None | 15 requests/min (free) |
| **Maintenance** | Update models periodically | Same | None |

---

## Option 1: Self-host at IQS

### What you need
- **A machine with an NVIDIA GPU** — a workstation or small server with at
  least a T4, RTX 3060 (12 GB), or better
- **NVIDIA drivers + CUDA** installed
- **Ollama** for local LLM summarization (replaces Gemini API dependency)
- **Docker** (recommended) or a Python environment
- **Network access** — either a public IP, or behind the IQS firewall with
  access from the campus network

### Hardware cost estimates

| Hardware | Approx. cost | Notes |
|---|---|---|
| Refurbished workstation + RTX 3060 12GB | ~500-800 EUR | Cheapest, runs full pipeline with quantized LLM |
| Refurbished workstation + RTX 3090 24GB | ~1,200-1,800 EUR | Best bang for buck, can run 70B LLM for top quality |
| NVIDIA Jetson Orin (64 GB) | ~1,500-2,000 EUR | Low power, purpose-built for inference |
| Dell/HP server + T4 GPU | ~2,000-4,000 EUR | Enterprise-grade, overkill unless shared |

### Running it

```bash
# Docker (recommended)
docker run --gpus all -p 7860:7860 catalan-lecture-processor

# Or directly
pip install -r requirements.txt
python app.py --share  # or behind nginx reverse proxy
```

### The full self-contained stack (no internet needed for processing)

1. **Whisper** (transcription) — already local, runs on GPU
2. **NLLB 3.3B** (translation) — already local, runs on GPU
3. **Ollama + Llama 3.1 8B** (cleanup + summarization) — local, runs on GPU
4. **Gradio** (web UI) — serves on local network

No Gemini API key. No HuggingFace token. No external API calls whatsoever.
Students upload audio, everything processes on your machine, results come back.

### Pros
- **One-time cost** — no monthly bills, no API fees
- **Fully self-contained** — no internet needed for processing
- **Full control** — no external dependencies, no API keys needed
- **Local LLM** (Ollama + Llama 3) handles summarization + cleanup
- **No session timeouts** — always on
- **Data stays on campus** — important for privacy

### Cons
- **Someone has to maintain it** — updates, reboots, driver issues
- **Single point of failure** — if the machine dies, service goes down
- **Upfront hardware cost**
- **Network setup** — need IT to open ports or set up reverse proxy
- **Electricity cost** — a GPU server draws 200-400W

### Who does this suit?
Best if IQS has a spare machine with a decent GPU, or is willing to buy one.
Ideal for a small department that wants full control and has some IT support.

---

## Option 2: BSC (Barcelona Supercomputing Center)

### What we'd ask for
- A small persistent allocation on their GPU cluster
- A way to expose a Gradio web UI (public URL or VPN)
- ~50 GB storage for models

### Why BSC would be interested
- This project uses **Projecte AINA's Catalan Whisper model** — built by BSC
- It directly supports Catalan language accessibility in higher education
- It's a showcase of their models being used in production
- Cristina (former BSC employee) is the contact point

### Pros
- **Free** — academic/research use
- **Powerful hardware** — MareNostrum has A100s, much faster than T4
- **Aligns with their mission** — Catalan language technology
- **No maintenance burden** on IQS side
- **Can host local LLMs** for summarization

### Cons
- **Depends on BSC approval** — not guaranteed
- **Bureaucracy** — may take time to set up
- **Less control** — their infrastructure, their rules
- **Network access** — may need VPN or special setup for student access
- **May not allow persistent web services** — would need to confirm

### Who does this suit?
Best option if BSC says yes. Worth pursuing in parallel with other options.

---

## Option 3: Cloud hosting (pay-as-you-go)

### Cheapest GPU cloud options

| Provider | GPU | Cost/hr | Cost/mo (24/7) | Cost/mo (8h/day weekdays) |
|---|---|---|---|---|
| **Vast.ai** | RTX 3090 | ~0.15-0.30 EUR | ~110-220 EUR | ~25-50 EUR |
| **RunPod** | RTX 3090 | ~0.25 EUR | ~180 EUR | ~40 EUR |
| **Lambda Labs** | A10 | ~0.50 EUR | ~360 EUR | ~80 EUR |
| **HuggingFace Spaces** | T4 | ~0.37 EUR | ~270 EUR | ~60 EUR |
| **GCP** | T4 | ~0.30 EUR | ~220 EUR | ~50 EUR |
| **AWS** | g4dn.xlarge (T4) | ~0.50 EUR | ~360 EUR | ~80 EUR |

### Smart cost-saving strategies
- **Auto-sleep**: Spin down GPU when no one is using it (HF Spaces does this)
- **Scheduled hours**: Only run during class hours (e.g. 8am-6pm weekdays)
- **Spot/preemptible instances**: 60-80% cheaper, but can be interrupted
- **Queue system**: Students submit jobs, GPU processes them in batch

### Pros
- **No hardware to buy or maintain**
- **Scale up/down as needed** — add GPUs during exam season
- **Professional uptime** — redundancy, monitoring
- **Easy HTTPS/domain setup**

### Cons
- **Recurring monthly cost** — EUR 25-80/month minimum for reasonable usage
- **Data leaves campus** — may have privacy implications
- **Vendor lock-in risk**
- **Cold starts** if using auto-sleep

### Who does this suit?
Good if IQS has a small budget (EUR 50-100/month) and wants zero hardware
maintenance. HuggingFace Spaces with a dedicated T4 is the simplest option.

---

## Option 4: HuggingFace Spaces (dedicated GPU)

This deserves its own section because it's the **simplest deployment path**.

### Setup
1. Create a HuggingFace account
2. Create a new Space with "Gradio" SDK
3. Upload `app.py` + `requirements.txt` (adapt from notebook)
4. Select "T4 medium" hardware (~$0.40/hr)
5. Enable auto-sleep after 15 min idle
6. Share the URL with students

### Cost estimate
- Light use (a few students/week): ~EUR 5-15/month (mostly sleeping)
- Moderate use (daily during semester): ~EUR 30-50/month
- Heavy use (many concurrent students): ~EUR 80-100/month

### Pros
- **Simplest setup** — no Docker, no servers, no IT department needed
- **Students just get a URL** — zero setup on their end
- **Auto-sleep saves money** when idle
- **Built-in versioning** via Git
- **HTTPS and domain included**

### Cons
- **Monthly cost** (though manageable)
- **Single GPU** — students queue if multiple use it at once
- **Cold start** after sleep (~2-3 min to reload models)
- **No local LLM** — would still need Gemini API for summarization
- **72h link expiry** does not apply (dedicated spaces are persistent)

---

## Option 5: Keep using Google Colab (current)

### What it is
Students open the notebook on Colab, run it, get a temporary public link.

### Pros
- **Free**
- **Already working**
- **No infrastructure to maintain**

### Cons
- **Students must set it up every time** (runtime, play button, wait)
- **Sessions time out** (504 errors when link dies)
- **No Gemini key = no summarization** (students would need their own key)
- **GPU availability not guaranteed** during peak times
- **Not a "service"** — it's a DIY tool

### Who does this suit?
Fine as a temporary solution while setting up something better. Good enough
for tech-savvy students, frustrating for others.

---

## Recommendation: phased approach

### Phase 1 — Now (free)
Keep using **Colab** with the faster-whisper notebook. Create a clear
step-by-step guide for students (already in GUIDE_FOR_STUDENTS.pdf).
Get a **Gemini API key** to enable summarization.

### Phase 2 — This semester (EUR 0-50/month)
Either:
- **BSC hosting** (if Cristina can help — free, best long-term)
- **HuggingFace Spaces** dedicated T4 (~EUR 30/month — simplest paid option)
- **Self-host** (if IQS has a spare GPU machine — one-time cost)

### Phase 3 — Next academic year
- Add local LLM for summarization (removes Gemini dependency)
- Add source language selection (Spanish, etc.)
- Add student authentication
- Add batch processing for multiple lectures

---

## Quick decision matrix

| Priority | Best option |
|---|---|
| **Cheapest (free)** | Colab (current) or BSC |
| **Easiest for students** | HuggingFace Spaces dedicated T4 |
| **Most control** | Self-host at IQS |
| **Best long-term** | BSC or self-host |
| **Fastest to set up** | HuggingFace Spaces |
| **No recurring costs** | Self-host or BSC |
| **Data stays on campus** | Self-host |
