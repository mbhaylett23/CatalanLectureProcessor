"""Gradio web interface for the Catalan Lecture Processor."""

import os
import re
import gradio as gr

from core.config import TARGET_LANGUAGES
from core.auth import get_user_storage_id
from core.pipeline import LectureProcessor, OUTPUT_DIR


def _get_user_history(username: str) -> list[list]:
    """List a user's past processing jobs for the history table."""
    user_dir = os.path.join(OUTPUT_DIR, username) if username else OUTPUT_DIR
    if not os.path.isdir(user_dir):
        return []

    rows = []
    for folder in sorted(os.listdir(user_dir), reverse=True):
        folder_path = os.path.join(user_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        # Parse date and name from folder: YYYY-MM-DD_audio_name
        parts = folder.split("_", 1)
        date = parts[0] if parts else folder
        name = parts[1].replace("_", " ") if len(parts) > 1 else folder
        # List available files
        files = [f for f in os.listdir(folder_path) if not f.startswith(".") and f != "desktop.ini"]
        has_zip = any(f.endswith(".zip") for f in files)
        file_summary = f"{len(files)} files" + (" (ZIP available)" if has_zip else "")
        rows.append([date, name, file_summary, folder])
    return rows


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

        with gr.Tabs():
            # ==============================================================
            # Tab 1: Process New Lecture
            # ==============================================================
            with gr.Tab("Process Lecture"):
                # -- Input section --
                with gr.Group():
                    gr.Markdown("### Upload Audio")
                    audio_input = gr.Audio(
                        label="Lecture audio (m4a, mp3, wav, ogg, webm, flac)",
                        type="filepath",
                        sources=["upload"],
                    )
                    upload_status = gr.Markdown(visible=False)
                    target_langs = gr.CheckboxGroup(
                        choices=TARGET_LANGUAGES,
                        value=["Spanish", "English"],
                        label="Translate to:",
                    )
                    gemini_api_key = gr.Textbox(
                        label="Optional Gemini API key",
                        type="password",
                        placeholder="Used for this run only. Falls back to NLLB if blank or unavailable.",
                        visible=False,
                    )
                    process_btn = gr.Button(
                        "Process Lecture", variant="primary", size="lg"
                    )

                    def on_audio_change(audio):
                        if audio:
                            name = os.path.basename(audio)
                            return gr.update(
                                value=f"**File loaded:** {name} — ready to process.",
                                visible=True,
                            )
                        return gr.update(value="", visible=False)

                    audio_input.change(on_audio_change, inputs=[audio_input], outputs=[upload_status])

                # -- Status --
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False, lines=3,
                    elem_id="status-box",
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

            # ==============================================================
            # Tab 2: My Lectures (History)
            # ==============================================================
            with gr.Tab("My Lectures"):
                gr.Markdown("### Processing History")
                gr.Markdown("Your previously processed lectures. Click a row to download files.")

                history_table = gr.Dataframe(
                    headers=["Date", "Lecture", "Files", "folder_id"],
                    column_count=(4, "fixed"),
                    interactive=False,
                    label="Past Lectures",
                )
                refresh_btn = gr.Button("Refresh", size="sm")

                with gr.Group():
                    gr.Markdown("### Download")
                    selected_label = gr.Textbox(
                        label="Selected Lecture", interactive=False
                    )
                    history_download = gr.File(
                        label="Files",
                        file_count="multiple",
                    )

        # -- Get username from Gradio auth --
        def _get_username(request: gr.Request) -> str:
            if request and request.username:
                return get_user_storage_id(request.username)
            return ""

        # -- Main processing function (generator for live UI updates) --
        def process_lecture(audio, languages, gemini_key, request: gr.Request):
            if audio is None:
                raise gr.Error("Please upload an audio file first.")
            if not languages:
                raise gr.Error("Please select at least one target language.")

            username = _get_username(request)
            processor = get_processor()

            # Check if this audio was already processed today
            audio_base = os.path.splitext(os.path.basename(audio))[0]
            safe_name = re.sub(r"[^\w\-]", "_", audio_base).strip("_")
            from datetime import datetime
            folder_name = f"{datetime.now().strftime('%Y-%m-%d')}_{safe_name}"
            user_dir = os.path.join(OUTPUT_DIR, username) if username else OUTPUT_DIR
            existing = os.path.join(user_dir, folder_name)
            if os.path.isdir(existing):
                gr.Warning("This audio was already processed today. Reprocessing will overwrite the previous results.")

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
                username=username,
                gemini_api_key=gemini_key,
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
            inputs=[audio_input, target_langs, gemini_api_key],
            outputs=all_outputs,
        )

        # -- History tab functions --
        def load_history(request: gr.Request):
            username = _get_username(request)
            rows = _get_user_history(username)
            return gr.update(value=rows if rows else [])

        def select_lecture(evt: gr.SelectData, table_data, request: gr.Request):
            """When user clicks a row, load files for download."""
            try:
                if table_data is None:
                    return "", None

                # Gradio 6 passes table_data as dict with "headers" and "data"
                if isinstance(table_data, dict):
                    rows = table_data.get("data", [])
                elif hasattr(table_data, 'values'):
                    # pandas DataFrame
                    rows = table_data.values.tolist()
                else:
                    rows = list(table_data)

                if not rows:
                    return "", None

                # evt.index is [row, col] in Gradio 6
                row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
                row = rows[row_idx]

                import logging
                logging.info("select_lecture debug: evt.index=%s, type(table_data)=%s, type(row)=%s, row=%s",
                             evt.index, type(table_data).__name__, type(row).__name__, row)

                # row might be a dict or a list
                if isinstance(row, dict):
                    folder_id = row.get("folder_id", "")
                else:
                    folder_id = row[3] if len(row) > 3 else str(row[-1])
                username = _get_username(request)
                user_dir = os.path.join(OUTPUT_DIR, username) if username else OUTPUT_DIR
                folder_path = os.path.join(user_dir, folder_id)

                if not os.path.isdir(folder_path):
                    return f"Folder not found: {folder_id}", None

                files = []
                for f in sorted(os.listdir(folder_path)):
                    if f.startswith(".") or f == "desktop.ini":
                        continue
                    files.append(os.path.join(folder_path, f))

                label = f"{row[0]} — {row[1]}"
                return label, files if files else None
            except Exception as e:
                return f"Error: {e}", None

        refresh_btn.click(load_history, inputs=[], outputs=[history_table])
        app.load(load_history, inputs=[], outputs=[history_table])

        history_table.select(
            select_lecture,
            inputs=[history_table],
            outputs=[selected_label, history_download],
        )

    return app
