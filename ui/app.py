"""Gradio web interface for the Catalan Lecture Processor."""

import os
import gradio as gr

from core.pipeline import LectureProcessor
from core.config import TARGET_LANGUAGES


def create_app(mode: str = "auto") -> gr.Blocks:
    """Create the Gradio Blocks application.

    Args:
        mode: "colab" (GPU + Gemini), "desktop" (CPU + Ollama), or "auto".
    """
    # Resolve mode
    if mode == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    elif mode == "colab":
        device = "cuda"
    else:
        device = "cpu"

    llm_backend = "auto"

    # Lazy-init processor on first use to avoid slow startup
    _processor = {}

    def get_processor() -> LectureProcessor:
        if "instance" not in _processor:
            _processor["instance"] = LectureProcessor(
                device=device, llm_backend=llm_backend
            )
        return _processor["instance"]

    with gr.Blocks(title="Catalan Lecture Processor") as app:

        gr.Markdown(
            "# Catalan Lecture Processor\n"
            "Upload a lecture recording in Catalan. Get transcription, "
            "translation, summary, and PowerPoint slides."
        )

        # -- Input section --
        with gr.Group():
            gr.Markdown("### Upload Audio")
            audio_input = gr.Audio(
                label="Lecture audio (m4a, mp3, wav, ogg, webm, flac)",
                type="filepath",
                sources=["upload"],
            )
            target_langs = gr.CheckboxGroup(
                choices=TARGET_LANGUAGES,
                value=["Spanish", "English"],
                label="Translate to:",
            )
            process_btn = gr.Button(
                "Process Lecture", variant="primary", size="lg"
            )

        # -- Status (moved above outputs for visibility) --
        status_text = gr.Textbox(
            label="Status", interactive=False, lines=3
        )
        errors_text = gr.Textbox(
            label="Warnings", interactive=False, lines=2, visible=False
        )

        # -- Output section (tabbed) --
        with gr.Tabs():
            with gr.Tab("Transcript"):
                transcript_raw = gr.Textbox(
                    label="Raw Transcription (Catalan)",
                    lines=12,
                    max_lines=30,
                )
                transcript_clean = gr.Textbox(
                    label="Cleaned Transcription",
                    lines=12,
                    max_lines=30,
                )

            with gr.Tab("Translations"):
                translation_boxes = {}
                for lang in TARGET_LANGUAGES:
                    translation_boxes[lang] = gr.Textbox(
                        label=f"{lang} Translation",
                        lines=10,
                        visible=(lang in ["Spanish", "English"]),
                    )

            with gr.Tab("Summary"):
                summary_output = gr.Markdown(label="Lecture Summary")

            with gr.Tab("Downloads"):
                download_zip = gr.File(
                    label="Download All (ZIP)",
                    file_count="single",
                )
                gr.Markdown("Or download individual files:")
                download_files = gr.File(
                    label="Individual Files",
                    file_count="multiple",
                )

        # -- Show/hide translation boxes based on selection --
        def update_translation_visibility(languages):
            return [
                gr.update(visible=(lang in languages))
                for lang in TARGET_LANGUAGES
            ]

        target_langs.change(
            update_translation_visibility,
            inputs=[target_langs],
            outputs=list(translation_boxes.values()),
        )

        # -- Main processing function (generator for live UI updates) --
        def process_lecture(audio, languages):
            if audio is None:
                raise gr.Error("Please upload an audio file first.")
            if not languages:
                raise gr.Error("Please select at least one target language.")

            processor = get_processor()

            def build_output(status, results):
                """Build the full output tuple from pipeline results."""
                raw = results.get("transcript_raw") or ""
                clean = results.get("transcript_clean") or ""
                translations = results.get("translations", {})
                trans_outputs = [
                    translations.get(lang, "") for lang in TARGET_LANGUAGES
                ]
                summary = results.get("summary") or ""
                zip_path = results.get("zip_path")
                files_list = list(results.get("all_files", {}).values()) or None
                errors = results.get("errors", [])
                errors_str = "\n".join(errors) if errors else ""

                return (
                    status,
                    gr.update(value=errors_str, visible=bool(errors)),
                    raw,
                    clean,
                    *trans_outputs,
                    summary,
                    zip_path,
                    files_list,
                )

            # Iterate over pipeline generator - each yield updates the UI
            for status, results in processor.process(
                audio_path=audio,
                target_languages=languages,
            ):
                yield build_output(status, results)

        all_outputs = [
            status_text,
            errors_text,
            transcript_raw,
            transcript_clean,
            *list(translation_boxes.values()),
            summary_output,
            download_zip,
            download_files,
        ]

        process_btn.click(
            process_lecture,
            inputs=[audio_input, target_langs],
            outputs=all_outputs,
        )

    return app
