# Audio Tool Development and Acceptance Standard

- Status: Development baseline
- Date: 2026-08-11
- Owners: Capicove maintainers
- Scope: Mandarin Chinese, English, and Mandarin-English code-switching

## Purpose

This document aligns product, model, backend, Agent-runtime, QA, and release work
for Capicove audio Tools. It defines the acceptance framework for:

- automatic speech recognition (ASR);
- text-to-speech synthesis (TTS);
- voice activity detection (VAD);
- speaker diarization;
- denoising and speech enhancement;
- CPU and GPU execution profiles; and
- transport-neutral Tool contracts for Agent use.

The individual audio operations are required. A combined `audio.pipeline`
operation is an enhancement and is not required for the first release.

## Normative language

The terms **must**, **must not**, **required**, **should**, and **may** describe
requirement strength. A requirement marked **must** is release-blocking unless an
approved capability-boundary declaration explicitly excludes the scenario before
testing begins.

All numeric thresholds in this document are recommendations, not universal
industry standards. Final release gates must be calibrated against an approved,
frozen, license-cleared business dataset. A metric is valid only when bound to the
dataset, hardware, model artifact, precision or quantization mode, runtime, and
evaluation configuration used to produce it. Teams must not reject a release by
applying a recommended table value without this calibration.

## Ownership and acceptance boundaries

| Area | Primary owner | Component release blocker |
| --- | --- | --- |
| Audio quality and correctness | Audio/model team | Yes |
| Canonical schemas and error semantics | Audio/runtime team | Yes |
| Capability discovery | Audio/runtime team | Yes |
| Security, privacy, and lineage | Audio/runtime team | Yes |
| Metrics, logs, and traces | Audio/runtime team | Yes |
| LLM Tool selection | Agent runtime | No |
| LLM argument generation | Agent runtime | No, unless the schema is defective |
| End-to-end Agent task success | Joint ownership | Joint release decision |

An Agent-runtime failure is attributed to the audio component when the Tool name,
description, schema, error classification, or capability declaration is incorrect
or ambiguous. An LLM selection failure against a correct and complete contract is
not by itself an audio-component release blocker.

## Bound acceptance profile

Every acceptance result must record:

- frozen dataset identifier, version, content hash, origin, and license;
- CPU model and core allocation, or GPU model and memory;
- operating system, driver, CUDA, and inference-runtime versions;
- model name, artifact version, and content hash;
- FP32, FP16, BF16, INT8, INT4, or other execution precision;
- decoding parameters, batch size, concurrency, and stream chunk size;
- input sample rate, channel count, codec, and preprocessing;
- enabled VAD, denoising, hotword, language-model, and post-processing stages;
- warm-up method, run count, and percentile calculation method; and
- evaluation code version and configuration hash.

Changing the model, quantization, runtime, material decoding behavior, or hardware
profile requires a new bound baseline. A release decision considers, in order:

1. security, data integrity, and contract blockers;
2. regression against the approved bound baseline;
3. agreed targets on the frozen business dataset; and
4. the recommended targets in this document as guidance.

## Three-level test strategy

### Smoke tests

Run smoke tests after a local implementation, image build, deployment change, and
before release packaging.

- Use 10 to 30 fixed samples and approximately 5 to 15 minutes of audio.
- Exercise the core path once on CPU and once on GPU when both are supported.
- Include Mandarin, English, and code-switching ASR and TTS samples.
- Include speech, silence, pure noise, two alternating speakers, and one ordinary
  noisy sample.
- Exercise file, streaming, malformed-input, capability-query, forced-GPU, and
  permitted-fallback paths.
- Verify that metrics, logs, traces, warnings, and errors are emitted.
- Target a total run time of 5 to 15 minutes.

Smoke tests verify availability, contracts, and obvious quality failures. They do
not include human MOS, AB, or ABX studies.

### CI regression tests

Run CI regression tests on relevant pull requests and model or runtime changes.

- Use a stratified set of approximately 100 to 500 samples.
- Cover critical languages, formats, noise conditions, durations, and error paths.
- Compare CER, WER, VAD F1, DER, objective enhancement metrics, and TTS
  transcription-back accuracy with the approved baseline.
- Check P50, P95, and when stable enough P99 latency on dedicated performance
  runners.
- Check RTF, memory, schema compatibility, error codes, determinism, timestamp
  monotonicity, capability accuracy, and output completeness.
- Do not require network access in the pull-request fast path.

Human MOS, human AB or ABX, full accent studies, 24-hour soak tests, and exhaustive
extreme-condition combinations do not run in ordinary CI. They belong to scheduled
or full-release evaluation.

### Full release acceptance

Run full acceptance for the first release, major releases, core-model replacement,
material quantization or inference changes, new languages or voices, new hardware
profiles, and security-boundary changes.

Full acceptance includes the complete frozen business suite, public benchmarks,
challenge sets, CPU/GPU matrices, human listening studies, subgroup analysis,
long-audio tests, a 24-hour soak, peak concurrency, cancellation and overload,
Agent-runtime joint tests, privacy review, voice authorization review, upgrade, and
rollback exercises.

## Shared evaluation matrix

Each required capability must cover the applicable dimensions below.

| Dimension | Required conditions |
| --- | --- |
| Language | Mandarin, English, Mandarin-English code-switching |
| Accent | Regional Mandarin; US, UK, and at least one non-native English accent |
| Demographic coverage | Adult women and men, older speakers, and children where in scope |
| Format | WAV, FLAC, MP3, and every format declared by the implementation |
| Sample rate | 8, 16, 24, 44.1, and 48 kHz where declared |
| Channels | Mono, stereo, and declared multichannel layouts |
| Duration | Sub-second, sentence, 5-30 minutes, and 1-4 hours |
| SNR | Clean, 20, 10, 5, 0, and -5 dB, subject to declared boundaries |
| Noise | Street, vehicle, office, keyboard, wind, music, and competing speech |
| Room | Close-talk, far-field, reverberant, and playback capture |
| Conversation | Single speaker, turns, rapid switches, interruptions, and overlap |
| Content | Numbers, money, dates, abbreviations, names, terms, URLs, and email |
| Execution | CPU and single GPU; batch and streaming where supported |
| Concurrency | 1, 4, 8, and the declared production peak |
| Invalid input | Empty, silent, corrupt, oversized, wrong MIME, OOM, and cancellation |

The evaluation corpus has four roles:

1. public benchmarks for reproducibility;
2. a frozen business set that is excluded from training and tuning;
3. a challenge set for noise, accents, overlap, short speech, and code-switching;
4. a stable regression set for model and runtime changes.

Recommended public references include LibriSpeech for English ASR, AISHELL-1 for
Mandarin ASR, AISHELL-3 for Mandarin TTS coverage, AISHELL-4 for Mandarin meetings,
NIST DIHARD-style DER/JER evaluation, and the Microsoft DNS Challenge with ITU-T
P.835 methodology for enhancement. Public clean corpora do not replace the frozen
business set.

## ASR acceptance

### Required behavior

- Support file transcription and streaming transcription when streaming is
  declared.
- Return complete text, segments, monotonic start and end timestamps, model
  lineage, and structured warnings.
- Produce stable partial and final states for streaming requests.
- Define punctuation, English casing, and inverse text normalization behavior.
- Cover numbers, money, dates, percentages, names, URLs, and business terms.
- Support Mandarin, English, and declared code-switching behavior.
- Make hotwords or prompts explicit and disableable.
- Optionally return word timestamps, confidence, and detected language when those
  outputs have documented semantics.
- Avoid missing, duplicated, reordered, or fabricated segments in long audio.
- Avoid long hallucinated transcripts for silence and pure noise.

Mandarin uses character error rate (CER) as the primary lexical metric. English
uses word error rate (WER):

```text
CER = (character substitutions + deletions + insertions) / reference characters
WER = (word substitutions + deletions + insertions) / reference words
```

### Recommended calibration targets

| Condition | Mandarin CER | English WER |
| --- | ---: | ---: |
| Clean close-talk | <= 8% | <= 8% |
| Aggregate business set | <= 12% | <= 12% |
| Noisy or far-field challenge set | <= 18% | <= 18% |
| Code-switching | <= 15% on Mandarin spans | <= 15% on English spans |

Additional recommended targets are at least 95% accuracy for numbers, money, and
dates; at least 95% recall for approved business terms; no more than one absolute
percentage point of ordinary-word degradation after hotwords are enabled; median
word-timestamp error at or below 150 ms and P95 at or below 500 ms; and hallucinated
content in no more than 0.5% of silence or pure-noise files.

For streaming, calibrate first-partial and finalization latency separately. Initial
guidance is P95 first partial at or below 500 ms on GPU and 1,000 ms on CPU, with
finalization at or below 1,000 ms on GPU and 2,000 ms on CPU after the endpoint.

Report error metrics by language, accent, age group, acoustic condition, and
content type. An aggregate pass must not hide a material regression in a critical
business subgroup.

## TTS acceptance

### Required behavior

- Support Mandarin, English, and declared mixed-language synthesis.
- Expose voice, rate, pitch, and volume controls with bounded semantics.
- Support streaming output when declared.
- Define pronunciation behavior for numbers, money, dates, units, abbreviations,
  URLs, Mandarin polyphones, and English letter sequences.
- Segment long text without omissions, repetitions, or broken words.
- Produce the requested supported sample rate, codec, and channel layout.
- Enforce voice authorization and return verifiable provenance.

### Recommended calibration targets

| Metric | Recommended target |
| --- | ---: |
| Human naturalness MOS, 1-5 | >= 3.8 and not below the approved baseline |
| Human intelligibility MOS | >= 4.0 |
| Independent-ASR CER on Mandarin output | <= 5% |
| Independent-ASR WER on English output | <= 5% |
| Numbers, dates, and money pronunciation | >= 98% accuracy |
| Code-switching content accuracy | >= 95% |
| Long-text omitted or repeated sentence rate | <= 0.5% |
| GPU first-audio P95 | <= 500 ms |
| CPU first-audio P95 | <= 1,000 ms |
| GPU RTF | <= 0.2 |
| CPU RTF | <= 1.0 |

Human MOS and AB or ABX evaluation runs only during full release acceptance. Voice
similarity must combine a blinded human study and a fixed, independent speaker
verification model. A universal cosine threshold is not valid across embedding
models; calibrate against natural recordings and an approved baseline.

### Verifiable voice provenance

Every synthesis result must reference a verifiable provenance record:

```json
{
  "voice_provenance": {
    "voice_id": "voice_zh_001",
    "voice_version": "3.2",
    "provider": "internal",
    "authorization_reference": "consent-record-reference",
    "model_artifact_hash": "sha256:...",
    "generated_at": "2026-08-11T12:00:00Z",
    "request_id": "request-id",
    "signature": "signature-value"
  }
}
```

The authorization reference must be verifiable without exposing sensitive consent
data to ordinary callers. Altering signed metadata must fail validation. Provenance
must survive transcoding through a signed sidecar or external manifest; a resilient
watermark may complement but must not replace canonical lineage. If watermarking is
used, test detection and false-positive rates after transcoding, cropping, and
noise. Revoked voices must be rejected for new requests.

## VAD acceptance

### Required behavior

- Return millisecond start and end times for speech regions.
- Emit streaming speech-start and speech-end events when streaming is supported.
- Expose minimum speech, minimum silence, and padding configuration.
- Preserve weak onsets, plosives, and quiet sentence endings within the declared
  latency-quality tradeoff.
- Define treatment of breathing, coughing, keyboard noise, music, and distant
  competing speech.
- Run for hours without timestamp drift or sustained memory growth.

### Recommended calibration targets

| Metric | Clean set | Challenge set |
| --- | ---: | ---: |
| Frame-level macro F1 | >= 0.95 | >= 0.90 |
| Speech miss rate | <= 3% | <= 8% |
| False-alarm rate | <= 3% | <= 8% |
| Start-boundary P95 | <= 200 ms | <= 300 ms |
| End-boundary P95 | <= 300 ms | <= 500 ms |
| Pure-noise activation | <= 1/hour | <= 3/hour |

Report duration-weighted miss and false-alarm rates, short-word recall, music
false-positive rate, endpoint latency, and the effect of VAD on downstream CER and
WER.

## Speaker diarization acceptance

### Required behavior

- Estimate the speaker count and accept an optional declared count or upper bound.
- Return stable speaker labels and monotonic segment timestamps.
- Maintain speaker identity across long recordings within declared limits.
- Report whether overlapping speech is detected, attributed, separated, ignored,
  or unsupported.
- Align speaker segments with ASR output when both results are requested.
- Support registered-speaker matching only when explicitly configured and
  authorized; otherwise return anonymous local speaker labels.
- Avoid repeatedly splitting one speaker after short pauses.

DER reporting must decompose missed speech, false-alarm speech, and speaker
confusion. Every report must fix and disclose overlap handling, collar width,
speaker mapping, and reference-annotation policy.

### Recommended calibration targets

| Condition | DER | JER |
| --- | ---: | ---: |
| Two speakers, close-talk, little overlap | <= 10% | <= 15% |
| Three to six speakers in a meeting | <= 15% | <= 22% |
| Far-field, reverberant, with overlap | <= 25% | <= 35% |

Additional guidance is speaker-count error zero for at least 90% of files and P95
absolute error at or below one; speaker-attributed or concatenated-permutation WER
at or below 20% on the calibrated business meeting set; speaker-change boundary
P95 at or below 500 ms; at least 90% recall for utterances shorter than five
seconds; and no more than 5% erroneous speaker splitting over 30 minutes.

Overlap detection F1 of at least 0.75 is an initial target only when overlap support
is declared. A system may declare overlap attribution or separation unsupported,
but it must expose that boundary through capability discovery and must not silently
return ordinary single-speaker results as complete.

## Denoising and speech-enhancement acceptance

Enhancement must preserve speech, not merely reduce audible noise. Human evaluation
should follow an ITU-T P.835-style separation of speech signal (SIG), background
noise (BAK), and overall quality (OVRL). Objective evaluation should combine
perceptual estimates, intrusive measures when clean references exist, and downstream
ASR behavior.

### Recommended calibration targets

- DNSMOS P.835 OVRL improvement of at least 0.30.
- BAK improvement of at least 0.40.
- SIG degradation no worse than 0.10.
- SI-SDR improvement of at least 3 dB when a clean reference exists.
- No STOI degradation, with a provisional target of at least 0.90.
- At least 10% relative downstream CER or WER reduction on the calibrated noisy set,
  without more than one absolute percentage point degradation in a critical noise
  category.
- No material metallic artifacts, musical noise, pumping, dropouts, or clipping.
- No removal of the target speaker during double-talk or competing-speech tests.
- Additional streaming processing latency at or below 40 ms when real-time
  enhancement is declared.

For high-SNR or nearly clean input, improvement is not required. The acceptance
condition is non-degradation of intelligibility, speech quality, downstream ASR,
and clipping behavior. Human listening tests are randomized and blinded, use both
Mandarin and English material, and report the listener count and 95% confidence
interval.

## Difficult-condition exemptions

Exemptions must be declared before evaluation and returned by capability discovery.
Permitted examples include sustained multi-speaker overlap, SNR below the validated
lower bound, unintelligible or annotation-disputed speech, nearly clean denoising
input, corrupt or clipped media, unsupported languages or codecs, excessive
speaker count, excessive duration, or excessive concurrency.

Exempt samples remain in the report with the rule, reason, and count. They are not
deleted after results are known. Low-SNR results must include a degradation curve.
Overlap results must be separated into overlap, non-overlap, and aggregate metrics.
If exemptions exceed the approved proportion of the business set, product scope
must be reviewed. Hard-boundary input is rejected; best-effort input returns a
structured warning.

## CPU and GPU execution profiles

CPU support means usable execution, not merely successful model loading.

| Requirement | CPU | GPU |
| --- | --- | --- |
| Installation | No implicit CUDA dependency | Published driver/CUDA matrix |
| Model profile | Quantized or smaller model permitted | FP16/BF16 and batching permitted |
| Contract | Same result and error semantics | Same result and error semantics |
| Quality | Calibrated; recommended <= 10% relative degradation from GPU | Quality reference profile |
| Online performance | Recommended RTF <= 1 | Recommended RTF <= 0.5 |
| Resources | Peak and steady RAM, leak test | Peak VRAM, fragmentation, OOM recovery |

Both profiles report cold start, model load, first result, P50/P95/P99 latency, RTF,
throughput, utilization, peak RAM or VRAM, supported concurrency, and 24-hour
stability. Quality comparisons use identical audio, decoding, normalization, and
pipeline configuration.

### Device policy

Forced GPU mode uses:

```json
{"device": "cuda", "fallback": "disabled"}
```

GPU absence, incompatibility, model-load failure, or OOM must return a stable error
such as `GPU_UNAVAILABLE`, `MODEL_LOAD_FAILED`, or `GPU_OOM`. It must not use CPU.

Permitted fallback uses:

```json
{"device": "cuda", "fallback": "cpu"}
```

A fallback must return `DEVICE_FALLBACK_APPLIED`, the requested and actual device,
the reason, and the actual-device service level. It must be observable and
cancellable. If `device: auto` is offered, selection order, resource thresholds,
and fallback policy must be public.

## Agent-facing Tool contract

The initial required operations are conceptually:

```text
audio.transcribe
audio.synthesize
audio.detect_voice
audio.diarize
audio.denoise
audio.get_job
audio.cancel_job
audio.get_capabilities
```

Canonical Capicove Tool IDs and transport names require normal contract and ADR
review before implementation. The required operations must support independent,
stepwise use. Large inputs use authorized resource references rather than forcing
base64 content into Agent context.

Each Tool must provide a stable name and version, machine-readable JSON Schema,
bounded arguments, examples, capability discovery, structured warnings, stable
errors, request idempotency, asynchronous jobs for long work, cancellation,
timeouts, model and backend lineage, and explicit partial-success semantics.

Recommended error codes include `INVALID_AUDIO`, `UNSUPPORTED_FORMAT`,
`UNSUPPORTED_LANGUAGE`, `OUTSIDE_VALIDATED_RANGE`, `MODEL_UNAVAILABLE`,
`GPU_UNAVAILABLE`, `GPU_OOM`, `MODEL_LOAD_FAILED`, `TIMEOUT`, and `CANCELLED`.
Errors must not expose server paths, credentials, stack traces, or user content.

### Capability declaration

`audio.get_capabilities` must report the actual loaded deployment configuration,
not static marketing metadata. A representative shape is:

```json
{
  "component": "audio.diarize",
  "version": "1.3.0",
  "model": {
    "name": "example-diarization-model",
    "version": "2026-08-01",
    "precision": "fp16",
    "artifact_hash": "sha256:..."
  },
  "execution": {
    "devices": ["cpu", "cuda"],
    "streaming": false,
    "fallback_policy": "explicit"
  },
  "supported": {
    "languages": ["zh-CN", "en-US"],
    "sample_rates_hz": [16000, 48000],
    "channels": [1, 2],
    "formats": ["wav", "flac"],
    "max_audio_duration_seconds": 14400,
    "speaker_count": {"min": 1, "max": 8},
    "overlapping_speech": "detected_but_not_fully_attributed"
  },
  "limitations": [
    {
      "condition": "snr_db < -5",
      "behavior": "best_effort",
      "warning_code": "LOW_SNR_OUT_OF_VALIDATED_RANGE"
    },
    {
      "condition": "speaker_count > 8",
      "behavior": "reject",
      "error_code": "UNSUPPORTED_SPEAKER_COUNT"
    }
  ],
  "validated_profile": {
    "dataset_version": "business-audio-eval-2026.08",
    "hardware_profile": "nvidia-l4-24gb",
    "benchmark_report_id": "audio-eval-2026-08-11"
  }
}
```

Documentation, capability output, and runtime behavior must agree. Hard-limit
violations are rejected. Best-effort conditions return warnings. Declarations are
versioned and covered by compatibility tests.

## Observability acceptance

Each request must make the following structured fields available to the authorized
observability system:

- request, trace, and span identifiers;
- Tool name and version;
- model name, version, artifact hash, precision, and actual device;
- input duration, sample rate, channels, and codec;
- queue, preprocessing, inference, post-processing, and total duration;
- first-result latency, RTF, and output duration;
- CPU/RAM or GPU/VRAM peak usage;
- result status, error codes, and warning codes; and
- retries, fallback, cancellation, and timeout state.

Operators must be able to aggregate success, timeout, OOM, fallback, queue depth,
and P50/P95/P99 latency by Tool, model, device, language, and error class. One trace
must cross the gateway, job queue, and inference service. Telemetry failure must not
fail inference. Logs must not contain raw audio, full transcripts, TTS input text,
credentials, or high-cardinality user identifiers by default. Audio and text use
controlled hashes or redacted evidence references.

## Optional combined pipeline

`audio.pipeline` is an enhancement. The required equivalent remains explicit
stepwise invocation:

```text
denoise -> VAD -> diarization -> ASR
```

If implemented, the pipeline must allow each stage to be configured or skipped,
return per-stage status, model lineage, timing, warnings, and evidence, preserve
authorized intermediate references, and map all timestamps back to the original
audio. It must support fail-fast and preserve-partial-result policies. The combined
result must be reproducible through the corresponding individual operations.

## Security and data governance

- Network access, remote inference, model download, and telemetry are explicit
  opt-ins.
- Source evidence, content hashes, versions, configuration, and lineage are
  preserved immutably.
- Temporary content follows a documented retention and deletion policy.
- Raw audio and full text are excluded from ordinary logs.
- Resource references are authorized and protected against path traversal and
  server-side request forgery.
- Model, dataset, and voice licenses are recorded and verified.
- Speaker recognition and voice cloning require explicit authorization.
- Source data, user content, credentials, model weights, and generated indexes are
  not committed to the repository.

## Release blockers

An audio-component release is blocked by any of the following:

- a required operation cannot install, start, or complete on a declared profile;
- frozen business-set quality regresses beyond the approved tolerance;
- security, privacy, authorization, evidence, or lineage fails;
- capability declarations disagree with runtime behavior;
- unsupported or best-effort conditions fail silently;
- forced GPU mode falls back to CPU;
- permitted fallback omits a structured warning and actual-device metadata;
- long audio loses, duplicates, or reorders content, or timestamps move backward;
- silence or noise produces hallucinations beyond the calibrated limit;
- denoising materially harms intelligibility or downstream ASR;
- TTS omits content or produces unverifiable voice provenance;
- the test dataset, model, quantization, hardware, or configuration is not
  reproducible; or
- required interface observability and audit evidence are absent.

The following do not independently block the audio component:

- an LLM selects the wrong Tool against a correct contract;
- a predeclared unsupported scenario is rejected correctly;
- an exempt extreme condition misses an ordinary-condition recommendation;
- high-SNR enhancement shows no objective improvement but causes no degradation;
  or
- a recommended value is missed when an approved, bound business baseline defines
  a different release threshold.

## Acceptance report template

Every full acceptance report must include:

```text
Release candidate:
Evaluator and date:
Dataset IDs, versions, hashes, origins, and licenses:
Model names, versions, hashes, and precision:
Runtime, OS, driver, and CUDA versions:
CPU/GPU hardware profiles:
Configuration hash:
Capability declaration version:

Smoke result:
CI regression result:
Full acceptance result:

ASR results by language and condition:
TTS objective and human results:
VAD results by condition:
Diarization results, collar, and overlap policy:
Enhancement SIG/BAK/OVRL, objective, and downstream results:
Latency, RTF, throughput, memory, and concurrency:
Observability and security results:
Agent-runtime joint-test results:

Exemptions and counts:
Known limitations:
Regressions from the approved baseline:
Failed sample references:
Waivers and approvers:
Final decision and approvers:
```

Reports must retain per-sample results, aggregate statistics, subgroup results,
failed-sample references, environment metadata, and the delta from the previous
approved baseline. An average score alone is insufficient evidence.

## Developer definition of done

Before requesting review for an audio capability, the developer must:

- add a failing test before production behavior;
- update the canonical schema and capability declaration;
- add or update smoke, contract, and relevant regression fixtures;
- record fixture origin and license without committing user or source content;
- test CPU and GPU behavior or document the declared profile boundary;
- test forced-device, fallback, cancellation, timeout, and invalid-input paths;
- add lineage, warning, error, metric, log, and trace coverage;
- update the bound baseline when a qualifying artifact or profile changes; and
- run `make quality`, `make test`, and `make build` before declaring completion.

## Primary references

- [LibriSpeech](https://www.openslr.org/12/)
- [AISHELL-1 Mandarin ASR corpus](https://openslr.org/33/)
- [AISHELL-3 Mandarin TTS corpus](https://www.openslr.org/93/)
- [AISHELL-4 Mandarin meeting corpus](https://www.openslr.org/111/)
- [NIST DIHARD III](https://sat.nist.gov/dihard3)
- [ITU-T P.835](https://www.itu.int/rec/T-REC-P.835-200311-I)
- [Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge)
