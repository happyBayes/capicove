"""Check the standalone TTS smoke command without loading real weights."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "tts_smoke.py"
    spec = importlib.util.spec_from_file_location("tts_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("platform", ["linux", "windows", "darwin"])
def test_platform_command_uses_local_files(platform: str, tmp_path: Path) -> None:
    command = load_script().build_command(tmp_path, platform, "Hello", None, tmp_path / "out.wav")
    assert "-hf" not in command
    assert command[command.index("-t") + 1] == "2"
    assert command[command.index("-ngl") + 1] == ("99" if platform == "darwin" else "0")
    assert command[0].endswith(".exe") == (platform == "windows")


def test_empty_text_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_script().build_command(tmp_path, "darwin", " ", None, tmp_path / "out.wav")
