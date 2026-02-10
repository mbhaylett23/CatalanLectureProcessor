# Catalan Lecture Processor - Student Guide

## What is this?

You record your university lectures in Catalan. This tool does everything else automatically:

1. **Listens** to your recording and writes down every word (transcription)
2. **Cleans up** the text - removes all the "umm", "ehh", "o sigui" that people say when speaking
3. **Translates** the text into Spanish, English, Portuguese, or Italian (you pick which ones)
4. **Summarises** the lecture into key points and main ideas
5. **Creates a PowerPoint** with slides based on the summary

You upload one audio file. You get back five things: the raw text, the cleaned text, translations, a summary, and a PowerPoint. All for free.

---

## How does it actually work? (No coding knowledge needed)

Think of it like a factory assembly line. Your audio recording goes in one end, and finished documents come out the other. Each station on the line does one job:

### Station 1: Transcription (Whisper)

**What it is:** Whisper is a program made by OpenAI (the same company that makes ChatGPT). It listens to audio and writes down what it hears - like a very fast, very accurate typist.

**Why it's special for you:** Normal Whisper already understands Catalan, but we use a version that was specifically trained on 710 hours of Catalan speech by researchers in Barcelona (Projecte AINA at the Barcelona Supercomputing Center). It's like the difference between someone who studied Catalan in a textbook vs. someone who grew up hearing it every day.

**What it needs:** This is the hardest job in the whole process. Listening to audio and understanding speech requires a lot of computing power. That's why we use Google's free computers (more on that below).

### Station 2: Text Cleanup

**What it is:** When people talk, they say things like "umm", "ehh", "doncs", "o sigui", "vull dir" - these are called filler words. Everyone does it. The cleanup station removes all of these automatically.

**How it works:** It uses two approaches:
- **Pattern matching** (always works) - like doing "Find and Replace" in Microsoft Word, but smarter. It knows all the common Catalan filler words and removes them while being careful not to accidentally damage real words.
- **AI polishing** (optional bonus) - if available, an AI reads through the text and reorganises it into neat paragraphs, fixes any weird sentences, and makes it read more like a written document than a spoken one.

### Station 3: Translation (NLLB-200)

**What it is:** NLLB stands for "No Language Left Behind." It's a translation program made by Meta (the company behind Facebook and Instagram). It can translate between over 200 languages.

**Why we use it:** It's completely free, works offline, and handles Catalan to Spanish, English, Portuguese, and Italian all in one program. You don't need four separate translators - one does everything.

**How good is it?** It's very good for getting the meaning across. It won't sound like a professional human translator wrote it, but it will be perfectly understandable and accurate enough for studying. Think of it as a very competent study buddy who speaks all these languages.

### Station 4: Summarisation

**What it is:** An AI reads through the entire cleaned transcript and pulls out:
- The **main topics** covered in the lecture (5-10 bullet points)
- A **detailed summary** (2-3 paragraphs)
- **Key terms** and important vocabulary

**Why this helps:** Instead of reading through pages of transcript, you can quickly see what the lecture was about. Great for revision or for deciding which parts you need to study in more detail.

### Station 5: PowerPoint Slides

**What it is:** The program automatically creates a `.pptx` file (PowerPoint presentation) from the summary. It includes:
- A title slide
- An overview of topics
- One or more slides per major topic
- A slide with key terms

**What you can do with it:** Open it in PowerPoint, Google Slides, or Keynote. Edit it, add your own notes, use it for studying, or share it with classmates.

---

## The two ways to use it

### Option A: Google Colab (Recommended - easiest and fastest)

**What is Google Colab?**
Imagine Google lets you borrow one of their powerful computers for free. That's basically what Colab is. You open a webpage, click a button, and Google's computer does all the heavy work. Your own computer just shows you the results.

**Why use it:**
- It's **fast** - a 1-hour lecture takes about 10-20 minutes to process
- It works from your **phone** - you can upload the recording right after class
- **Nothing to install** - just a web browser and a Google account
- It's **free** - Google provides the computing power at no cost

**The catch:**
- You need internet
- Google doesn't guarantee the powerful computer is always available (it usually is, but during very busy times you might have to wait)
- Your session ends after about 12 hours of inactivity

**How to use it (step by step):**

1. Open your web browser (Chrome, Safari, Firefox - any will work)
2. Go to [colab.research.google.com](https://colab.research.google.com)
3. Sign in with your Google account
4. Click **File** → **Upload notebook**
5. Select the file `lecture_processor.ipynb` from this project's `colab/` folder
6. At the top, click **Runtime** → **Change runtime type**
7. Select **T4 GPU** from the dropdown and click **Save**
8. Click **Runtime** → **Run all**
9. Wait 2-3 minutes. You'll see text scrolling as it sets things up.
10. At the very bottom, a **link** will appear that looks like `https://xxxxx.gradio.live`
11. **Click that link** (or copy it to your phone!)
12. You'll see the app. Upload your audio, pick your languages, click **Process Lecture**
13. Wait for it to finish, then download your files

**Pro tip:** You can open that link on your phone right after class, upload the recording, and have everything ready by the time you get home.

### Option B: Desktop App (For offline use)

**What is it?**
The same tool, but running on your own computer instead of Google's. Good for when you don't have internet or want to work privately.

**Why use it:**
- Works **offline** (after initial setup)
- No time limits
- Your recordings stay on your computer
- Has a **desktop icon** - just double-click to launch like any other app

**The catch:**
- **Slower** - a 1-hour lecture might take 30-60 minutes on a modern computer
- The first time you run it, it needs to download some things (~5 minutes on good wifi)
- Text cleanup and summarisation work better with an extra free program called Ollama (optional)

---

#### One-time setup (you only do this once)

You'll need someone to help you with these steps the first time. After that, you just double-click the icon.

**Step 1: Install Python** (the language the app is written in)

- **Mac:** Open the **Terminal** app (search for it in Spotlight) and paste:
  ```
  brew install python@3.12 ffmpeg
  ```
  (If `brew` isn't installed, first paste this:)
  ```
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

- **Windows:** Download Python from [python.org](https://www.python.org/downloads/) and install it.
  **IMPORTANT: tick the box that says "Add Python to PATH" during installation.**
  Then open Command Prompt and type:
  ```
  winget install ffmpeg
  ```

**Step 2: Create the desktop icon**

- **Windows:** Double-click `create_shortcut_windows.bat` in the project folder. A shortcut called **"Catalan Lecture Processor"** will appear on your desktop.

- **Mac:** Double-click `create_shortcut_mac.command` in the project folder. An app called **"Catalan Lecture Processor"** will appear on your desktop.
  - macOS may ask you to allow it: go to **System Settings > Privacy & Security** and click **Allow**.

**Step 3 (optional): Install Ollama** for smarter text cleanup and summaries

- **Mac:** Open Terminal and paste: `brew install ollama && ollama pull llama3.2:3b`
- **Windows:** Download from [ollama.com](https://ollama.com/download/windows), install it, then open Command Prompt and type: `ollama pull llama3.2:3b`

That's it for setup!

---

#### How to use it every day

1. **Double-click the desktop icon** ("Catalan Lecture Processor")
2. The first time, it will spend a few minutes setting things up (you'll see a progress screen). This only happens once.
3. Your web browser opens automatically with the app
4. Upload your audio, select languages, click **Process Lecture**
5. When you're done, just close the window

That's it. No terminal, no commands, no coding. Just double-click and go.

---

## What audio formats work?

You can upload recordings in any of these formats:
- **.m4a** - what iPhones record in (this is probably what you have)
- **.mp3** - the most common audio format
- **.wav** - high quality, larger files
- **.ogg** - common on Android phones
- **.webm** - web format
- **.flac** - high quality compressed

**You don't need to convert anything.** Just upload whatever your phone records.

---

## What do I get back?

After processing, you can download all of these:

| File | What it is |
|------|-----------|
| `transcript_raw.txt` | Everything that was said, word for word (in Catalan) |
| `transcript_clean.txt` | Same text but with filler words removed and tidied up |
| `translation_spanish.txt` | The lecture translated to Spanish |
| `translation_english.txt` | The lecture translated to English |
| `translation_portuguese.txt` | The lecture translated to Portuguese |
| `translation_italian.txt` | The lecture translated to Italian |
| `summary.md` | A summary with main topics, detailed overview, and key terms |
| `lecture_slides.pptx` | A PowerPoint presentation based on the summary |

You only get translations for the languages you selected. Everything is downloadable from the **Downloads** tab.

---

## Tips for best results

1. **Recording quality matters.** Try to sit near the front. A clearer recording = better transcription.

2. **Quieter is better.** Background noise (other students talking, fans, etc.) makes it harder for Whisper to understand.

3. **Use your phone's voice recorder app.** Most phones have one built in. Just hit record at the start of the lecture.

4. **For very long lectures (2+ hours):** Colab is strongly recommended. Desktop mode will work but will be slow.

5. **The summary and slides are only as good as the transcription.** If the audio quality is poor, the transcription will have errors, and those errors carry through to everything else.

6. **You can always edit the results.** Open the transcript or PowerPoint and fix anything that doesn't look right. Think of this as a 90% solution that saves you hours of work.

---

## Frequently asked questions

**Q: Does this cost anything?**
A: No. Everything is completely free. No subscriptions, no credit cards, no hidden fees.

**Q: Does anyone else hear my recordings?**
A: On Colab, the audio is processed on Google's computers (similar to uploading to Google Drive). On desktop mode, everything stays on your computer. Neither option shares your data with anyone.

**Q: What if the transcription has mistakes?**
A: It will have some mistakes, especially with technical terminology, proper names, or if the audio is noisy. The Catalan-specific model is very good, but no transcription is perfect. Think of it as a very good first draft.

**Q: Can I use this for languages other than Catalan?**
A: The tool was built specifically for Catalan lectures. The transcription model and filler word removal are Catalan-specific. However, the translation and summarisation parts could work with other languages with some modifications.

**Q: What if Colab says "GPU not available"?**
A: This happens occasionally during busy times. You have three options:
1. Wait 10-15 minutes and try again
2. Try [Kaggle](https://www.kaggle.com) instead (similar free service - you'd need to adapt the notebook slightly)
3. Use the desktop app

**Q: Do I need to understand Python or coding?**
A: Not at all for the Colab option. Just follow the steps above - it's all clicking buttons. The desktop setup needs a tiny bit of terminal use, but once it's set up you just run one command.

---

## Glossary of technical terms (in case you're curious)

| Term | What it means |
|------|--------------|
| **Whisper** | An AI program that converts speech to text. Like having a super-fast typist who can understand almost any language. |
| **NLLB-200** | "No Language Left Behind" - a translation AI by Meta that knows 200+ languages. |
| **GPU** | Graphics Processing Unit - a special computer chip that's really good at the kind of maths AI needs. Normally used for video games, but perfect for AI too. |
| **Google Colab** | A free service where Google lends you one of their computers (with a GPU) through your web browser. |
| **Gradio** | A tool that creates simple web pages for AI programs. It's what makes the nice upload-and-click interface you see. |
| **Ollama** | A free program that runs AI language models on your own computer. Like having a small ChatGPT that works offline. |
| **Gemini** | Google's AI (like ChatGPT but by Google). We can use its free version for text cleanup. |
| **Python** | A programming language. You don't need to know it - it's just what the tool is written in. |
| **ffmpeg** | A free program that converts audio/video between different formats. Works behind the scenes. |
| **PowerPoint (.pptx)** | A presentation file format. Can be opened in Microsoft PowerPoint, Google Slides, or Apple Keynote. |
| **Transcription** | Converting spoken words into written text. |
| **Filler words** | The "umm", "ehh", "so basically" sounds people make when speaking. Not useful in written text. |
