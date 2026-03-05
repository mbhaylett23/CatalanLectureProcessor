# Deployment Options for Catalan Lecture Processor

Comparison of platforms for running the lecture processor, evaluated for a
university course where ~20-30 students need to transcribe 30-60 min Catalan
lectures using Whisper + NLLB translation on GPU.

**Key requirement:** Transcription of a 1-hour lecture needs ~5-10 minutes of
continuous GPU time (using faster-whisper on a T4-class GPU).

---

## Summary Table

| Platform | GPU | Cost | Student Setup | Max GPU Time | Verdict |
|---|---|---|---|---|---|
| **Google Colab (free)** | T4 (15 GB) | Free | Upload notebook, set runtime | 12h session | Best free option |
| **Google Colab Pro (edu)** | T4/V100/A100 | Free (US only) | Verify .edu email | Longer sessions | US students only |
| **Kaggle Notebooks** | P100 (16 GB) | Free | Create account | 30 hrs/week | Good alternative |
| **Lightning.ai** | T4/A10G | Free ~35 hrs/mo | Create account | Per-session limits | Viable |
| **HF Spaces (ZeroGPU)** | H200 (70 GB) | Free to use | Visit URL | 3.5 min/day (free) | Too limited |
| **HF Spaces (dedicated)** | T4 (16 GB) | $0.40/hr | Visit URL | Unlimited | Best UX, costs $ |
| **Azure for Students** | Various | $100 credit | Verify student email | Until credit runs out | Worth exploring |
| **Google Cloud (edu)** | Various | $50 credit | Faculty applies | Until credit runs out | Faculty must apply |
| **GitHub Codespaces** | CPU only | Free (students) | GitHub account | 180 hrs/mo | No GPU — unusable |

---

## Catalonia-Specific Resources

### CSUC (Consorci de Serveis Universitaris de Catalunya)

[CSUC](https://www.csuc.cat/en) is the consortium of all public universities
in Catalonia, three private universities, and the Generalitat. It provides
supercomputing, cloud infrastructure, and data management services.

- CSUC has procured GPU server infrastructure for its member universities
- **Worth asking:** Whether your university has access to CSUC GPU resources
  for teaching purposes — contact your institution's IT department
- This would be the ideal solution if available: institutional infrastructure,
  no cost to students, no external dependencies

### Barcelona Supercomputing Center (BSC-CNS)

[BSC](https://www.bsc.es/) houses MareNostrum and runs Projecte AINA — the
very team that created the `whisper-large-v3-ca-3catparla` Catalan Whisper
model used in the original notebook.

- **PATC Courses** (PRACE Advanced Training Centre): Free training courses
  with access to European supercomputing resources. Open to researchers and
  students. See [BSC Education](https://www.bsc.es/education)
- **PUMPS+AI Summer School**: Annual program with hands-on GPU access
- **Potential collaboration:** Since this project uses Projecte AINA's model
  for Catalan language processing in a university context, BSC/AINA may be
  interested in supporting it or providing compute access

**Source:** [Projecte AINA](https://projecteaina.cat/en/) |
[BSC GPU Center of Excellence](http://ccoe.ac.upc.edu/)

### Projecte AINA

[Projecte AINA](https://projecteaina.cat/en/) is a Generalitat de Catalunya +
BSC partnership to ensure Catalan thrives in the digital/AI era. They provide
open language models, datasets, and tools.

- They created the Catalan Whisper model this project uses
- The **AINA Challenge** has funded 22 AI+Catalan language projects (€1M total)
- **Worth contacting:** If this tool is being used at a Catalan university for
  Catalan language education, it aligns directly with AINA's mission. They may
  be able to provide compute resources or host the tool

**Contact:** [projecteaina.cat](https://projecteaina.cat/en/)

### NVIDIA Academic Grant Program

[NVIDIA's program](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/)
provides GPU hardware, cloud credits, or software to researchers worldwide.

- **Eligibility:** Faculty, post-docs, or grad students with faculty sponsor
  at any accredited institution worldwide (not US-only)
- **What you can get:** Up to 30,000 NVIDIA H100 hours, or physical RTX GPUs
- **Caveat:** Aimed at research, not classroom teaching. Would need to frame
  this as a research/innovation project
- **Apply:** [academicgrants.nvidia.com](https://academicgrants.nvidia.com/academicgrantprogram/s/Application)

---

## Cloud Provider Education Credits

These programs give cloud credits that can be used for GPU VMs, but require
more setup than Colab (launching VMs, configuring environments).

### Microsoft Azure for Students

- **Credit:** $100 USD, no credit card required
- **Eligibility:** Students 18+ at any accredited institution worldwide
- **GPU access:** Possible but may require requesting a quota increase for
  GPU VM sizes — not guaranteed
- **Duration:** 12 months
- **How:** Verify with institutional email at
  [azure.microsoft.com/free/students](https://azure.microsoft.com/en-in/free/students)

**Note:** $100 goes fast on GPU VMs (~$0.50-1.00/hr for T4 equivalent).
About 100-200 hours of GPU time.

### Google Cloud for Education

- **Credit:** $50 per student, $100 for faculty
- **Eligibility:** Faculty at eligible institutions applies, then distributes
  coupon codes to students. Available internationally (expanding countries).
- **GPU access:** Yes, via Compute Engine GPU VMs
- **How:** Faculty emails cloudedugrants@google.com or applies through
  [cloud.google.com/edu](https://cloud.google.com/edu)

**Note:** More setup overhead than Colab (need to create VMs, install deps).
Better suited if you want to teach cloud computing alongside the content.

### AWS Educate

- **Credit:** Varies, typically $50-100 per student
- **Eligibility:** Students and educators worldwide
- **GPU access:** Yes via EC2 GPU instances, but may require quota approval
- **How:** [aws.amazon.com/education/awseducate](https://aws.amazon.com/education/awseducate/)

---

## Detailed Platform Breakdown

### 1. Google Colab (Free Tier) — CURRENT APPROACH

- **GPU:** T4 (15 GB VRAM)
- **Cost:** Free
- **Session limit:** ~12 hours, may disconnect after 90 min idle
- **Setup:** Student uploads `.ipynb`, sets runtime to T4 GPU, clicks play
- **Pros:**
  - Free, no credit card needed
  - Familiar to many students
  - Our notebook already works here
- **Cons:**
  - Students must configure runtime manually ("Change runtime type > T4")
  - Sessions can disconnect if idle
  - GPU availability not guaranteed during peak times
  - Free tier may throttle heavy users

**Status:** Working. The `lecture_processor_faster_whisper.ipynb` notebook is
ready for this platform.

### 2. Google Colab Pro (Education) — US-ONLY

Google offers **free 1-year Colab Pro subscriptions** to students and faculty
at US-based higher education institutions.

- **GPU:** T4, V100, A100 (priority access to faster GPUs)
- **Cost:** Free for verified US students/faculty (normally $9.99/mo)
- **How to get it:** Visit [colab.research.google.com/signup](https://colab.research.google.com/signup)
- **Eligibility:** US-based university only, verified with institutional email
- **Renewal:** Can re-verify after 335 days

**Source:** [Google Blog — Colab for Higher Education](https://blog.google/products-and-platforms/products/education/colab-higher-education/)

### 3. Kaggle Notebooks

- **GPU:** NVIDIA P100 (16 GB VRAM)
- **Cost:** Free
- **Limit:** 30 hours/week of GPU time
- **Setup:** Create Kaggle account, upload notebook, enable GPU in settings
- **Pros:**
  - 30 hrs/week is very generous — enough for many lectures
  - P100 is comparable to T4 for inference
  - Free, no verification needed
- **Cons:**
  - Interface less polished than Colab
  - Would need to adapt the notebook slightly (Kaggle uses different paths)
  - Sharing a public link via Gradio may have networking restrictions
  - Less familiar to students than Colab

**Note:** The Gradio `share=True` public link may not work on Kaggle due to
network restrictions. Would need testing.

### 4. Lightning.ai

- **GPU:** T4, A10G (depending on availability)
- **Cost:** Free tier includes ~35 GPU hours/month (15 credits)
- **Setup:** Create account, create a "Studio", upload code
- **Pros:**
  - Generous free GPU allocation
  - Full Linux environment (not just notebooks)
  - Can run persistent apps
- **Cons:**
  - Less familiar platform for students
  - More complex setup than Colab
  - Would need to package the app differently
  - Free tier may change

### 5. HuggingFace Spaces — ZeroGPU (Shared)

- **GPU:** NVIDIA H200 (70 GB VRAM, allocated on-demand)
- **Cost:** Free to use (hosting requires HF PRO at $9/mo)
- **How it works:** GPU allocated when student clicks "Process", released after
- **Daily quotas:**
  - Unauthenticated: 2 min GPU/day
  - Free HF account: 3.5 min GPU/day
  - PRO ($9/mo): 25 min GPU/day
- **Pros:**
  - Students just visit a URL — zero setup
  - H200 is extremely fast
  - Best possible user experience
- **Cons:**
  - **Daily quota is too small** — a 1h lecture needs 5-10 min of GPU,
    exceeding the free 3.5 min daily limit
  - Per-function timeout (default 60s, extendable but limited)
  - Requires HF PRO to host ($9/mo)
  - Queue wait times during peak usage

**Verdict:** Great for short demos, impractical for full-length lectures on
free tier.

### 6. HuggingFace Spaces — Dedicated GPU

- **GPU:** T4 (16 GB), always on
- **Cost:** $0.40/hr (~$10/mo if 24/7, less with auto-sleep)
- **Setup for students:** Just visit a URL
- **Pros:**
  - **Best UX** — students open a link and it works, nothing to configure
  - No quotas, no timeouts, no queue
  - Can auto-sleep after idle period to save costs
  - Same T4 hardware as Colab
- **Cons:**
  - Costs money ($0.40/hr while running)
  - Instructor must manage the Space
  - Cold start when waking from sleep (~2-3 min)
  - Single instance — if many students use it simultaneously, they queue

**Estimated cost for a semester:** $5-30/month depending on usage patterns
and sleep settings.

### 7. GitHub Codespaces (Student Developer Pack)

- **What students get:** 180 Codespaces hours/month, Copilot Pro, various tools
- **GPU:** None — CPU only
- **Relevance:** Cannot run Whisper or NLLB efficiently without GPU

**Source:** [GitHub Education Student Developer Pack](https://education.github.com/pack)

**Verdict:** Great for general development, but **no GPU support** makes it
unusable for this project.

---

## Recommendation

### For a Catalan university (current situation):

1. **Best free option: Google Colab (free tier)** — already working, students
   are somewhat familiar with it, and the faster-whisper notebook minimizes
   GPU time and avoids hallucinations.

2. **Best UX (costs money): HuggingFace Spaces with dedicated T4** — students
   just get a URL. ~$10-15/month during the semester.

3. **Free alternative worth testing: Kaggle Notebooks** — 30 hrs/week is
   generous, but Gradio public links may not work (needs testing).

4. **Worth exploring:**
   - Ask your university IT about **CSUC GPU resources** for teaching
   - Contact **Projecte AINA / BSC** — this project directly supports Catalan
     language technology, which is their mission
   - Apply for **Google Cloud for Education** credits ($50/student)
   - Have students sign up for **Azure for Students** ($100 free credit each)

### For a US-based university:

1. **Best option: Google Colab Pro (edu)** — free for students, faster GPUs,
   longer sessions. Apply at colab.research.google.com/signup.

---

## Current Status

The project currently supports:
- **Colab:** `colab/lecture_processor_faster_whisper.ipynb` (faster-whisper + VAD)
- **Colab (legacy):** `colab/lecture_processor_simple.ipynb` (transformers pipeline)
- **Desktop:** `setup_and_run.py` with local Gradio UI

To add HuggingFace Spaces support, the faster-whisper notebook would need to
be adapted into an `app.py` with a `requirements.txt`.
