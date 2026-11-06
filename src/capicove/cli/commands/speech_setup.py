"""Opt-in onboarding for offline speech synthesis and reference voices."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the local speech installer."""
    parser = parent.add_parser(
        "setup-speech", help="Prepare local TTS and reference voice synthesis."
    )
    parser.add_argument("--root", type=Path, help="Speech directory (default: ~/.capicove/speech).")
    parser.add_argument(
        "--plan", action="store_true", help="Show the pinned download plan and exit."
    )
    parser.add_argument(
        "--yes", action="store_true", help="Confirm the exact downloads in the plan."
    )
    parser.add_argument("--assets-dir", type=Path, help="Use the two verified local GGUF files.")
    parser.add_argument(
        "--llama-tts", type=Path, help="Use an existing compatible local executable."
    )
    parser.add_argument("--reference", type=Path, help="A 3-30 second mono PCM16 WAV.")
    parser.add_argument(
        "--authorize-reference",
        action="store_true",
        help="Confirm you have permission to use this recording for voice synthesis.",
    )
    parser.add_argument("--test-text", help="Generate a local WAV after setup.")
    parser.add_argument("--language", default="zh", help="Language for the trial synthesis.")


def run(arguments: argparse.Namespace) -> int:
    """Confirm downloads and recording rights before provisioning the runtime."""
    from capicove.processing.audio.synthesis import SpeechRequest
    from capicove.runtime.speech import SpeechRuntime
    from capicove.runtime.speech_setup import (
        grant_reference,
        install_speech,
        read_reference,
        setup_lock,
        setup_plan,
    )

    root = (arguments.root or Path.home() / ".capicove" / "speech").expanduser().absolute()
    assets = arguments.assets_dir.expanduser().absolute() if arguments.assets_dir else None
    binary = arguments.llama_tts.expanduser().absolute() if arguments.llama_tts else None
    runtime = None
    try:
        plan = setup_plan(root, assets, binary)
        if arguments.plan:
            print(json.dumps(plan, indent=2))
            return 0
        if bool(arguments.reference) != arguments.authorize_reference:
            raise ValueError("--reference and --authorize-reference must be supplied together")
        reference_bytes = (
            read_reference(arguments.reference.expanduser().absolute())
            if arguments.reference
            else None
        )
        downloads = bool(plan["model_download"] or plan["backend_download"])
        if downloads:
            print(json.dumps(plan, indent=2), file=sys.stderr)
            if not arguments.yes:
                if not sys.stdin.isatty():
                    raise ValueError("review --plan and pass --yes to authorize these downloads")
                print("Type yes to authorize the downloads shown above:", file=sys.stderr)
                if input().strip().lower() != "yes":
                    raise ValueError("speech downloads were declined")
            print("Preparing speech assets. This may take several minutes.", file=sys.stderr)

        with setup_lock(root):
            install_speech(root, assets, binary, download_authorized=downloads)
            config = root / "speech.json"
            key = root / "signing.key"
            runtime = SpeechRuntime(config, signing_key_file=key)
            result: dict[str, object] = {
                "runtime_config": str(config),
                "model_config": str(root / "model.json"),
                "signing_key_file": str(key),
                "capabilities": runtime.capabilities(),
            }
            reference, grant = None, None
            if reference_bytes is not None:
                reference, grant = grant_reference(root, runtime, reference_bytes)
                result.update(reference=str(reference), authorization=grant)

        if arguments.test_text is not None:
            try:
                result["trial"] = runtime.generate(
                    SpeechRequest(
                        arguments.test_text,
                        arguments.language,
                        reference=reference,
                        authorization=grant,
                    ),
                    datetime.now(UTC) + timedelta(seconds=240),
                )
            except (OSError, ValueError) as error:
                result["trial_error"] = str(error)
                print(json.dumps(result, ensure_ascii=False))
                return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, EOFError, subprocess.SubprocessError) as error:
        print(f"Speech setup failed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Speech setup interrupted.", file=sys.stderr)
        return 130
    finally:
        if runtime is not None:
            runtime.close()
