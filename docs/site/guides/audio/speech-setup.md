# Set up local speech synthesis

Install Capicove and make CMake and a C++ toolchain available on your path. On
macOS, the Xcode Command Line Tools provide the compiler. On Windows, use Visual
Studio C++ Build Tools and CMake. You can also supply an existing compatible
llama-tts executable with `--llama-tts /absolute/path/to/llama-tts`.

Review the download plan, then start setup with an optional local trial:

```sh
capicove audio setup-speech --plan
capicove audio setup-speech --test-text "Hello." --language en
```

At the prompt, type `yes` to approve the displayed sources and usage terms.
For automation, review the plan first and use `--yes` as the explicit download
authorization. The source archive and both Qwen3-TTS GGUF files are pinned by
revision, byte length, and SHA-256. The two model files occupy about 2.52 GB.
Building the backend also needs temporary disk space. Setup selects Metal for
Apple Silicon and CPU for other supported devices.

To work from a source checkout, use the same options through the script:

```sh
uv run python scripts/setup_speech.py --plan
uv run python scripts/setup_speech.py --test-text "Hello." --language en
```

When you already have the executable and both exact GGUF files, pass
`--llama-tts` and `--assets-dir` together to configure them offline. The script
copies them into the managed directory and verifies the GGUF sizes and hashes.

The default directory is `~/.capicove/speech`. Use `--root` to select another
private location outside the repository. Setup returns JSON containing paths to
`model.json`, `speech.json`, and the owner-private `signing.key`. The key is used
to sign local output provenance. The acquisition record is `provisioning.json`.
Rerunning setup for the same directory verifies the existing installation and
preserves its key and grants. Select a new directory to use different files.

To authorize a reference WAV and generate a cloned trial, run:

```sh
capicove audio setup-speech --reference /absolute/path/to/reference.wav \
  --authorize-reference --test-text "Hello." --language en
```

Use a complete mono PCM16 WAV that is 3 to 30 seconds long and at most 16 MiB.
`--authorize-reference` affirms that you have the right to use this recording
for voice synthesis. Setup keeps a private copy and binds its exact SHA-256
to the authorization identifier. The result includes the reference path,
authorization identifier, and trial WAV path. Reuse these values with the
normal offline CLI. For example:

```sh
capicove audio synthesize --runtime-config /absolute/speech/speech.json \
  --signing-key-file /absolute/speech/signing.key --text "Hello." --language en

capicove audio clone --runtime-config /absolute/speech/speech.json \
  --signing-key-file /absolute/speech/signing.key --reference /absolute/managed-reference.wav \
  --authorization grant-from-setup --text "Hello." --language en
```

A failed trial returns exit code 1 and a `trial_error` alongside the installed
configuration paths. You can retry the trial against the same installation.
Setup and acquisition errors return exit code 2. For a new installation, an
acquisition or verification error leaves the destination unpublished.

The setup flow is a local development preview. Platform and real-model audio
qualification are tracked separately from installer and CLI tests.
