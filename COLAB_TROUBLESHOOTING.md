# Google Colab Troubleshooting Guide

> **Tip:** If you're having trouble with cells not running in order, try the **simplified notebook** (`lecture_processor_simple.ipynb`) — it has just ONE cell to run, so there's nothing to get out of order.

## Problem: "I can't see the link at the bottom"

The public link (like `https://xxxxx.gradio.live`) only appears after ALL cells finish running. Here's how to fix common issues:

---

### Solution 1: Scroll down to the last cell

The link appears in the output of the **very last cell** (Cell 11 - "Gradio UI and Launch"). You may need to scroll all the way to the bottom of the notebook. The cells above show setup output that you don't need to read.

---

### Solution 2: Not all cells ran

If you clicked "Run all" but see a red error on an early cell, the remaining cells may not have executed.

**Fix:** Scroll down to the last cell. If it doesn't show any output:
1. Click on the last cell (Cell 11)
2. Press **Shift+Enter** to run it manually
3. Or click **Runtime > Run all** again

---

### Solution 3: Cell 1 shows a red error (GPU check)

Cell 1 runs `!nvidia-smi` to check the GPU. This sometimes fails even when the GPU IS connected.

**Check:** Look at the bottom-right of the screen. If it says **"T4 (Python 3)"**, the GPU is fine. The Cell 1 error is harmless.

**Fix:** Ignore the error and run the remaining cells. Click on Cell 2 and press **Shift+Enter** for each cell, or click **Runtime > Run all** again.

---

### Solution 4: "Cannot connect to a GPU runtime" or "GPU not available"

This means Google's free GPU quota is exhausted.

**Fixes (try in order):**
1. Wait 15-30 minutes and try again
2. Try a different time of day (mornings European time are less busy)
3. Go to **Runtime > Change runtime type** and try **L4 GPU** instead of T4
4. Use a different Google account
5. Use the desktop app instead (slower but always available)

---

### Solution 5: Cells run but the link never appears

The last cell starts a Gradio server with `share=True` which creates a public tunnel. This can fail if:

**a) Gradio tunnel is blocked:**
- Some university/corporate networks block Gradio's tunneling service
- **Fix:** Try on a different network (phone hotspot, home wifi)

**b) The cell is still running:**
- Look for a spinning circle on the last cell - it means it's still loading
- The first run downloads ~3GB of models. Wait 3-5 minutes.

**c) The cell finished but no link appeared:**
- Look for error text in the cell output (red text)
- Screenshot it and send to your instructor

---

### Solution 6: "ModuleNotFoundError" or import errors

Cell 2 installs all dependencies. If it was skipped or failed:

**Fix:**
1. Click on Cell 2 (Install dependencies)
2. Press **Shift+Enter** to run it
3. Wait for it to finish (no output means it worked - `%%capture` hides output)
4. Then run the remaining cells

---

### Solution 7: Runtime disconnected / session expired

Colab disconnects after ~90 minutes of inactivity or ~12 hours total.

**Signs:** Cells show old output, clicking "Run" does nothing, or you see "Runtime disconnected".

**Fix:**
1. Click **Runtime > Reconnect** (or **Runtime > Disconnect and delete runtime**, then **Runtime > Run all**)
2. All cells need to run again from the beginning

---

### Solution 8: "No module named 'google.genai'" or Gemini errors

This only affects the summarization step (optional). The app still works without it.

**Fix:** Either:
- Ignore it (transcription and translation still work fine)
- Or paste a free Gemini API key in Cell 3 (get one at https://ai.google.dev/)

---

### Solution 9: Audio upload fails or processing hangs

**Possible causes:**
- File too large (Colab has a ~100MB upload limit through the Gradio interface)
- Unsupported format (use .m4a, .mp3, .wav, .ogg, .webm, or .flac)
- Network timeout during upload

**Fix:**
- Try a shorter audio clip first to test
- Convert to .mp3 if using an unusual format
- Upload from a stable wifi connection

---

### Solution 10: "NameError: name 'LectureProcessor' is not defined"

This error means the final cell (Cell 11) ran before the earlier cells that define the processing classes. Each cell builds on the previous ones — Cell 11 needs Cells 4–10 to have already run.

**Common causes:**
- You clicked directly on the last cell and pressed Shift+Enter, without running earlier cells first
- You clicked **Runtime > Run all** but an earlier cell had a red error that stopped execution
- The runtime was disconnected/restarted, clearing all previous cell outputs

**Fix:**
1. Click **Runtime > Run all** to run every cell from the beginning
2. Wait for each cell to finish — you should see `"Configuration loaded!"`, `"Transcriber ready"`, `"TextCleaner ready"`, `"Translator ready"`, `"Summarizer ready"`, `"SlideGenerator ready"`, and `"LectureProcessor ready"` printed in Cells 4–10
3. If any cell shows a red error, fix that first (see the other solutions above)
4. Only after all cells show green checkmarks should the last cell work

**Quick verification:** Scroll through Cells 4–10. Each should show a short confirmation message (e.g., `"Translator ready"`). If any cell is blank or shows an error, that's the cell you need to run first.

---

## Quick checklist before asking for help

1. Is "T4 (Python 3)" shown in the bottom-right? (GPU is connected)
2. Did you click **Runtime > Run all**?
3. Did you scroll to the very last cell?
4. Did you wait 3-5 minutes for models to download?
5. Is there a red error on any cell? (screenshot it)

If you've checked all of these, screenshot the **last cell's output** and send it to your instructor.
