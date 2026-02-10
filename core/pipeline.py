"""Pipeline orchestrator that chains transcription, cleanup, translation,
summarization, and slide generation."""

import os
import time
import tempfile
import logging

from core.config import SUPPORTED_AUDIO_FORMATS
from core.transcriber import Transcriber
from core.cleaner import TextCleaner
from core.translator import Translator
from core.summarizer import Summarizer
from core.slides import SlideGenerator

logger = logging.getLogger(__name__)

TOTAL_STEPS = 5


def _progress_bar(fraction, width=30):
    """Render a text-based progress bar like [████████░░░░░░░░░░░░] 40%"""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(width * fraction)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"[{bar}] {fraction * 100:.0f}%"


def _step(n, title, detail="", progress=None):
    """Format a step status string with optional progress bar."""
    header = f"Step {n}/{TOTAL_STEPS} \u2014 {title}"
    if progress is not None:
        header += f"  {_progress_bar(progress)}"
    if detail:
        return f"{header}\n{detail}"
    return header


class LectureProcessor:
    """End-to-end lecture processing pipeline.

    Orchestrates: transcribe -> clean -> translate -> summarize -> slides.
    Each step is independent; if one fails the rest continue.
    """

    def __init__(self, device: str = "auto", llm_backend: str = "auto"):
        self._device = device
        self._llm_backend = llm_backend
        self._transcriber = None
        self._cleaner = None
        self._translator = None
        self._summarizer = None
        self._slide_generator = SlideGenerator()

    def _get_transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber = Transcriber(device=self._device)
        return self._transcriber

    def _get_cleaner(self) -> TextCleaner:
        if self._cleaner is None:
            self._cleaner = TextCleaner(llm_backend=self._llm_backend)
        return self._cleaner

    def _get_translator(self) -> Translator:
        if self._translator is None:
            self._translator = Translator(device=self._device)
        return self._translator

    def _get_summarizer(self) -> Summarizer:
        if self._summarizer is None:
            self._summarizer = Summarizer(llm_backend=self._llm_backend)
        return self._summarizer

    def _validate_audio(self, audio_path: str):
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Supported: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
            )
        if os.path.getsize(audio_path) == 0:
            raise ValueError("Audio file is empty (0 bytes)")

    def _save_text(self, text: str, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

    def process(self, audio_path: str, target_languages: list[str], **kwargs):
        """Run the full processing pipeline as a generator.

        Yields (status_message, results_dict) tuples so the UI can show
        real-time progress with step numbers and progress bars.
        """
        results = {
            "transcript_raw": None,
            "transcript_clean": None,
            "translations": {},
            "summary": None,
            "summary_data": None,
            "slides_path": None,
            "all_files": {},
            "errors": [],
            "timings": {},
        }

        output_dir = tempfile.mkdtemp(prefix="lecture_")

        # -- Validate --
        yield ("Validating audio file...", results)
        try:
            self._validate_audio(audio_path)
        except (FileNotFoundError, ValueError) as e:
            results["errors"].append(str(e))
            yield (f"Error: {e}", results)
            return

        # ============================================================
        # Step 1: Transcribe
        # ============================================================
        yield (_step(1, "Transcribing", "Loading model... (this takes up to 30s)"), results)

        t0 = time.time()
        try:
            transcriber = self._get_transcriber()
            transcriber._ensure_loaded()

            if transcriber.device == "cpu" and transcriber._model is not None:
                # Streaming transcription with per-segment progress
                transcription = None
                for update in transcriber._transcribe_cpu_streaming(audio_path):
                    if update["done"]:
                        transcription = update["result"]
                    else:
                        yield (
                            _step(1, "Transcribing",
                                  update["desc"],
                                  progress=update["progress"]),
                            results,
                        )
                transcription["duration_seconds"] = time.time() - t0
            else:
                yield (_step(1, "Transcribing", "Using GPU..."), results)
                transcription = transcriber.transcribe(audio_path)

            results["transcript_raw"] = transcription["text"]
            results["timings"]["transcription"] = time.time() - t0

            raw_path = os.path.join(output_dir, "transcript_raw.txt")
            self._save_text(results["transcript_raw"], raw_path)
            results["all_files"]["transcript_raw.txt"] = raw_path

            logger.info(
                "Transcription completed in %.1f seconds",
                results["timings"]["transcription"],
            )
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            results["errors"].append(f"Transcription failed: {e}")
            yield (f"Transcription failed: {e}", results)
            return

        # ============================================================
        # Step 2: Clean text
        # ============================================================
        t0 = time.time()
        try:
            clean_result = None
            for update in self._get_cleaner().clean_streaming(
                results["transcript_raw"]
            ):
                if update["done"]:
                    clean_result = update["result"]
                else:
                    desc = update.get("desc", "")
                    progress = update.get("progress")
                    yield (
                        _step(2, "Cleaning text", desc, progress=progress),
                        results,
                    )

            results["transcript_clean"] = clean_result["best"]
            results["timings"]["cleanup"] = time.time() - t0

            clean_path = os.path.join(output_dir, "transcript_clean.txt")
            self._save_text(results["transcript_clean"], clean_path)
            results["all_files"]["transcript_clean.txt"] = clean_path
        except Exception as e:
            logger.error("Text cleanup failed: %s", e)
            results["errors"].append(f"Text cleanup failed: {e}")
            results["transcript_clean"] = results["transcript_raw"]

        # ============================================================
        # Step 3: Translate
        # ============================================================
        yield (_step(3, "Translating", "Loading model... (this takes up to 30s)"), results)

        t0 = time.time()
        text_to_translate = results["transcript_clean"] or results["transcript_raw"]
        try:
            translator = self._get_translator()
            translator._ensure_loaded()

            num_langs = len(target_languages)
            for i, lang in enumerate(target_languages):
                try:
                    for update in translator.translate_text_streaming(
                        text_to_translate, "Catalan", lang
                    ):
                        if update["done"]:
                            results["translations"][lang] = update["result"]
                        else:
                            # Combined progress: language + batch within language
                            lang_base = i / num_langs
                            lang_portion = 1 / num_langs
                            total_frac = lang_base + lang_portion * update["progress"]
                            detail = (
                                f"{lang} \u2014 batch {update['batch']}"
                                f"/{update['total']}"
                            )
                            yield (
                                _step(3, "Translating", detail,
                                      progress=total_frac),
                                results,
                            )

                    # Save translated file
                    translated = results["translations"].get(lang, "")
                    if translated and not translated.startswith("[Translation"):
                        filename = f"translation_{lang.lower()}.txt"
                        filepath = os.path.join(output_dir, filename)
                        self._save_text(translated, filepath)
                        results["all_files"][filename] = filepath
                except Exception as e:
                    logger.error("Translation to %s failed: %s", lang, e)
                    results["translations"][lang] = (
                        f"[Translation to {lang} failed: {e}]"
                    )

            results["timings"]["translation"] = time.time() - t0
        except Exception as e:
            logger.error("Translation failed: %s", e)
            results["errors"].append(f"Translation failed: {e}")

        # ============================================================
        # Step 4: Summarize
        # ============================================================
        t0 = time.time()
        text_to_summarize = results["transcript_clean"] or results["transcript_raw"]
        try:
            summary_data = None
            for update in self._get_summarizer().summarize_streaming(
                text_to_summarize, "Catalan"
            ):
                if update["done"]:
                    summary_data = update["result"]
                else:
                    desc = update.get("desc", "")
                    progress = update.get("progress")
                    yield (
                        _step(4, "Summarizing", desc, progress=progress),
                        results,
                    )

            results["summary"] = summary_data.get("raw_summary")
            results["summary_data"] = summary_data
            results["timings"]["summarization"] = time.time() - t0

            if results["summary"]:
                summary_path = os.path.join(output_dir, "summary.md")
                self._save_text(results["summary"], summary_path)
                results["all_files"]["summary.md"] = summary_path
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            results["errors"].append(f"Summarization failed: {e}")

        # ============================================================
        # Step 5: Generate slides
        # ============================================================
        yield (_step(5, "Creating slides"), results)

        t0 = time.time()
        try:
            summary_data = results.get("summary_data") or {}
            if summary_data.get("main_topics") or summary_data.get("sections"):
                slides_path = os.path.join(output_dir, "lecture_slides.pptx")
                audio_name = os.path.splitext(os.path.basename(audio_path))[0]
                self._slide_generator.generate(
                    summary_data, title=audio_name, output_path=slides_path
                )
                results["slides_path"] = slides_path
                results["all_files"]["lecture_slides.pptx"] = slides_path
                results["timings"]["slides"] = time.time() - t0
            else:
                results["errors"].append(
                    "Slides skipped: no summary data available"
                )
        except Exception as e:
            logger.error("Slide generation failed: %s", e)
            results["errors"].append(f"Slide generation failed: {e}")

        # Build timing summary
        timing_parts = [f"{k}: {v:.1f}s" for k, v in results["timings"].items()]
        status = "Done! " + " | ".join(timing_parts) if timing_parts else "Done!"

        if results["errors"]:
            logger.warning("Pipeline completed with errors: %s", results["errors"])
        else:
            logger.info("Pipeline completed successfully")

        yield (status, results)
