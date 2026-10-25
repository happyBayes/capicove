"""Install release wheels in an isolated environment outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


PROFILES = {
    "base": ("capicove", ""),
    "interfaces": ("capicove", "wiki,mcp,api"),
    "generation": ("capicove", "graph-remote"),
    "knowledge": ("capicove", "knowledge,mcp,api"),
    "spark": ("capicove-execution-spark", ""),
    "paddle": ("capicove-processor-paddleocr", ""),
    "elasticsearch": ("capicove-retriever-elasticsearch", ""),
    "opensearch": ("capicove-retriever-opensearch", ""),
    "faiss": ("capicove-retriever-faiss", ""),
}


def audit_requirements(installed: dict[str, str]) -> tuple[str, dict[str, dict[str, str]]]:
    """Map official CPU Torch builds to public advisory versions, retaining inventory."""
    local_packages = {distribution for distribution, _extras in PROFILES.values()}
    requirements = []
    aliases = {}
    for name, version in sorted(installed.items()):
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in local_packages:
            continue
        audited = version
        # The knowledge installer selects the official PyTorch CPU index. Its
        # local build suffix is absent from PyPI's advisory version inventory.
        if canonical == "torch" and re.fullmatch(r"\d+\.\d+\.\d+\+cpu", version):
            audited = version.removesuffix("+cpu")
            aliases[name] = {"installed": version, "audited": audited}
        requirements.append(f"{name}=={audited}")
    return "\n".join(requirements) + "\n", aliases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="interfaces")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--constraints", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    distribution, extras = PROFILES[args.profile]
    directory = args.dist.resolve()
    with tempfile.TemporaryDirectory(prefix="capicove-release-smoke-") as temporary:
        root = Path(temporary)
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "PYTHONPATH",
                "PYTHONHOME",
                "VIRTUAL_ENV",
                "UV_PROJECT_ENVIRONMENT",
                "UV_NO_SYNC",
                "UV_OFFLINE",
                "PIP_CONSTRAINT",
                "PIP_REQUIREMENT",
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
            }
        }
        environment.update(
            UV_CACHE_DIR=str(root / "cache"),
            HF_HUB_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            TRANSFORMERS_OFFLINE="1",
        )
        outputs: list[dict[str, object]] = []

        def run(command: list[str]) -> str:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=1200,
                check=False,
            )
            outputs.append({"command": command, "returncode": completed.returncode})
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
            return completed.stdout

        run(["uv", "venv", "--python", sys.executable, str(root / "venv")])
        scripts = root / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = str(scripts / ("python.exe" if os.name == "nt" else "python"))
        executable = str(scripts / ("capicove.exe" if os.name == "nt" else "capicove"))
        core = next(directory.glob("capicove-*-py3-none-any.whl"))
        wheels = [str(core) + (f"[{extras}]" if extras else "")]
        if distribution != "capicove":
            wheels.append(str(next(directory.glob(distribution.replace("-", "_") + "-*.whl"))))
        command = ["uv", "pip", "install", "--python", python, *wheels]
        if args.profile == "knowledge":
            command.extend(["--torch-backend", "cpu"])
        if args.constraints:
            command.extend(["--constraints", str(args.constraints.resolve())])
        run(command)
        run(["uv", "pip", "check", "--python", python])
        run(
            [
                python,
                "-I",
                "-c",
                "import capicove,pathlib,sys; assert pathlib.Path(capicove.__file__).is_relative_to(pathlib.Path(sys.prefix)); print(capicove.__file__)",
            ]
        )
        run([executable, "--version"])
        run([executable, "doctor"])
        image = root / "authored.ppm"
        image.write_text("P3\n2 2\n255\n255 0 0  0 255 0\n0 0 255  255 255 255\n")
        result = json.loads(
            run(
                [
                    executable,
                    "image",
                    "run",
                    "--tool",
                    "image_aspect_ratio_filter",
                    "--input",
                    str(image),
                    "--output",
                    "json",
                ]
            )
        )
        if result["status"] != "succeeded" or result["result"]["keep"] is not True:
            raise RuntimeError("installed image CLI failed its authored fixture")
        if args.profile == "interfaces":
            note = root / "notes" / "energy.md"
            note.parent.mkdir()
            note.write_text("# Energy\nSolar panels convert sunlight into electricity.\n")
            wiki = root / "wiki"
            run([executable, "wiki", "init", str(wiki)])
            run([executable, "wiki", "ingest", str(wiki), str(note)])
            run([executable, "wiki", "build", str(wiki)])
            if "energy" not in run([executable, "wiki", "query", str(wiki), "sunlight"]).lower():
                raise RuntimeError("installed wiki did not retrieve authored evidence")
            workspace = root / "workspace"
            run(
                [
                    executable,
                    "setup",
                    "init",
                    str(workspace),
                    "--source-root",
                    f"notes={note.parent}",
                ]
            )
            config = str(workspace / "host.json")
            run([executable, "setup", "doctor", "--config", config])
            run(
                [
                    executable,
                    "source",
                    "import",
                    "--runtime-config",
                    config,
                    "--root",
                    "notes",
                    "--relative-path",
                    note.name,
                ]
            )
            run([executable, "index", "build", "--runtime-config", config])
            run([executable, "search", "--runtime-config", config, "sunlight"])
            run([executable, "mcp", "catalog", "--config", config])
        elif args.profile == "generation":
            run(
                [
                    python,
                    "-I",
                    "-c",
                    "import importlib.util; from tokenizers import Tokenizer; "
                    "from tokenizers.models import WordLevel; "
                    "from transformers import PreTrainedTokenizerFast; "
                    "assert importlib.util.find_spec('torch') is None; "
                    "raw=Tokenizer(WordLevel({'[UNK]':0,'hello':1},unk_token='[UNK]')); "
                    "tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]'); "
                    "tokenizer.chat_template=\"{% for m in messages %}{{ m['content'] }}{% endfor %}\"; "
                    "ids=tokenizer.apply_chat_template([{'role':'user','content':'hello'}],"
                    "tokenize=True,return_dict=False); assert ids == [1]; print('offline chat template ready')",
                ]
            )
        elif args.profile == "knowledge":
            from deploy_release import qualify_knowledge

            run([python, "-I", "-c", "import torch,pypdf,pypdfium2,sentence_transformers"])
            qualify_knowledge(executable, root, run)
        elif args.profile != "base":
            code = (
                "from importlib.metadata import distribution; "
                f"d=distribution({distribution!r}); "
                "eps=[ep for ep in d.entry_points if ep.group.startswith('capicove.')]; "
                "assert eps; [ep.load()() for ep in eps]; print(d.version)"
            )
            run([python, "-I", "-c", code])
            if args.profile == "faiss":
                run(
                    [
                        python,
                        "-I",
                        "-c",
                        "import faiss,numpy as np; x=np.eye(2,dtype='float32'); i=faiss.IndexFlatL2(2); i.add(x); assert i.search(x,1)[1].ravel().tolist()==[0,1]",
                    ]
                )
        installed = json.loads(
            run(
                [
                    python,
                    "-I",
                    "-c",
                    "import json; from importlib.metadata import distributions; print(json.dumps({d.metadata['Name']:d.version for d in distributions()}, sort_keys=True))",
                ]
            )
        )
        artifacts = {
            Path(wheel.split("[", 1)[0]).name: hashlib.sha256(
                Path(wheel.split("[", 1)[0]).read_bytes()
            ).hexdigest()
            for wheel in wheels
        }
        aliases: dict[str, dict[str, str]] = {}
        if args.audit:
            inventory, aliases = audit_requirements(installed)
            requirements = root / "installed-requirements.txt"
            requirements.write_text(inventory, encoding="utf-8")
            run(
                [
                    sys.executable,
                    "-m",
                    "pip_audit",
                    "--requirement",
                    str(requirements),
                    "--no-deps",
                    "--disable-pip",
                    "--strict",
                    "--progress-spinner",
                    "off",
                ]
            )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "profile": args.profile,
                    "status": "passed",
                    "python": sys.version,
                    "checks": outputs,
                    "installed": installed,
                    "audit_version_aliases": aliases,
                    "artifacts_sha256": artifacts,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Clean wheel installation passed: {args.profile}")


if __name__ == "__main__":
    main()
