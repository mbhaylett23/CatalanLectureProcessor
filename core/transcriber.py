"""Speech-to-text transcription using Whisper (GPU) or faster-whisper (CPU)."""

import os
import time
import logging

from core.config import (
    WHISPER_HF_MODEL,
    WHISPER_FASTER_MODEL,
    WHISPER_FASTER_FALLBACK,
    WHISPER_BATCH_SIZE,
    WHISPER_BEAM_SIZE,
    WHISPER_CHUNK_LENGTH_S,
    WHISPER_VAD_PARAMS,
    SUPPORTED_AUDIO_FORMATS,
)

logger = logging.getLogger(__name__)


class Transcriber:
    """Transcribe audio files using Whisper models.

    Automatically selects GPU (transformers pipeline) or CPU (faster-whisper)
    based on available hardware.
    """

    def __init__(self, device: str = "auto", model_id: str = None):
        """
        Args:
            device: "auto" (detect), "cuda", or "cpu".
            model_id: Override default model selection.
        """
        self.device = self._resolve_device(device)
        self.model_id = model_id
        self._model = None
        self._pipe = None

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_gpu_model(self):
        """Load transformers pipeline with float16 on CUDA."""
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model_id = self.model_id or WHISPER_HF_MODEL
        logger.info("Loading GPU model: %s", model_id)

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda:0")

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda:0",
            chunk_length_s=WHISPER_CHUNK_LENGTH_S,
            batch_size=WHISPER_BATCH_SIZE,
            return_timestamps=True,
        )
        logger.info("GPU model loaded successfully")

    def _load_cpu_model(self):
        """Load faster-whisper model with INT8 quantization for CPU."""
        from faster_whisper import WhisperModel

        model_id = self.model_id or WHISPER_FASTER_MODEL
        logger.info("Loading CPU model: %s", model_id)

        try:
            self._model = WhisperModel(
                model_id, device="cpu", compute_type="int8"
            )
        except Exception as e:
            logger.warning(
                "Failed to load %s (%s), falling back to %s",
                model_id, e, WHISPER_FASTER_FALLBACK,
            )
            self._model = WhisperModel(
                WHISPER_FASTER_FALLBACK, device="cpu", compute_type="int8"
            )
        logger.info("CPU model loaded successfully")

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self.device == "cuda" and self._pipe is None:
            self._load_gpu_model()
        elif self.device == "cpu" and self._model is None:
            self._load_cpu_model()

    def transcribe(self, audio_path: str, progress_callback=None) -> dict:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            progress_callback: Optional callable(fraction, desc=str) for progress.

        Returns:
            dict with keys: text, segments, language, duration_seconds
        """
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Supported: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
            )

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if os.path.getsize(audio_path) == 0:
            raise ValueError("Audio file is empty (0 bytes)")

        if progress_callback:
            progress_callback(0.05, desc="Loading transcription model...")

        self._ensure_loaded()

        if progress_callback:
            progress_callback(0.10, desc="Transcribing audio...")

        start_time = time.time()

        if self.device == "cuda":
            result = self._transcribe_gpu(audio_path, progress_callback)
        else:
            result = self._transcribe_cpu(audio_path, progress_callback)

        result["duration_seconds"] = time.time() - start_time
        logger.info(
            "Transcription completed in %.1f seconds", result["duration_seconds"]
        )
        return result

    def _transcribe_gpu(self, audio_path: str, progress_callback=None) -> dict:
        """Transcribe using transformers pipeline on GPU."""
        output = self._pipe(
            audio_path,
            generate_kwargs={"language": "ca", "task": "transcribe"},
        )

        # Build segments from chunks
        segments = []
        if "chunks" in output:
            for chunk in output["chunks"]:
                ts = chunk.get("timestamp", (None, None))
                segments.append({
                    "start": ts[0] if ts[0] is not None else 0.0,
                    "end": ts[1] if ts[1] is not None else 0.0,
                    "text": chunk["text"].strip(),
                })

        if progress_callback:
            progress_callback(0.50, desc="Transcription complete")

        return {
            "text": output["text"].strip(),
            "segments": segments,
            "language": "ca",
        }

    def _transcribe_cpu(self, audio_path: str, progress_callback=None) -> dict:
        """Transcribe using faster-whisper on CPU (non-streaming)."""
        result = None
        for update in self._transcribe_cpu_streaming(audio_path):
            if update.get("done"):
                result = update["result"]
        return result

    def _transcribe_cpu_streaming(self, audio_path: str):
        """Generator that yields progress updates during CPU transcription.

        Yields dicts with keys:
            - progress: float 0.0 to 1.0
            - desc: human-readable status string
            - done: True on the final yield (includes 'result' key)
            - result: (only on final yield) dict with text, segments, language
        """
        from faster_whisper import BatchedInferencePipeline

        batched = BatchedInferencePipeline(model=self._model)
        segments_iter, info = batched.transcribe(
            audio_path,
            language="ca",
            batch_size=WHISPER_BATCH_SIZE,
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=True,
            vad_parameters=WHISPER_VAD_PARAMS,
        )

        segments = []
        full_text_parts = []
        total_duration = info.duration if info.duration else 1.0

        for seg in segments_iter:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

            frac = min(seg.end / total_duration, 1.0) if total_duration > 0 else 0
            yield {
                "progress": frac,
                "desc": f"{seg.end:.0f}s / {total_duration:.0f}s",
                "done": False,
            }

        full_text = " ".join(full_text_parts)

        yield {
            "progress": 1.0,
            "desc": "Transcription complete",
            "done": True,
            "result": {
                "text": full_text,
                "segments": segments,
                "language": "ca",
            },
        }

    def unload(self):
        """Free model memory."""
        self._model = None
        self._pipe = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        import gc
        gc.collect()
