# Blackboard LaTeX transcription

This fork adds a required handwritten-board transcription path for courses
whose title matches `BLACKBOARD_COURSES`. The default whitelist contains only
`泛函分析`; every other course continues to use the original ASR + PPT OCR +
summary pipeline and does not spend vision-model calls.

## Functional-analysis behavior

For a matching course, a lecture is not considered complete until the real
lecture video has produced a non-empty blackboard transcription. The runtime:

1. obtains the same signed iCourse MP4 used by the audio pipeline;
2. densely samples one video frame every 15 seconds by default;
3. performs **local temporal selection before any vision API call**;
4. keeps at least one relatively stable representative frame per minute across
   the whole lecture, plus persistent board-change events;
5. caps very long lectures at 240 selected vision frames by default while
   preserving timeline coverage first;
6. sends only those selected frames to the verified ModelScope
   `Qwen/Qwen3-VL-8B-Instruct` endpoint;
7. classifies projected slides/screens as `no_board` and transcribes genuine
   handwritten blackboard/whiteboard frames;
8. requires math to be emitted as LaTeX and uses `[unclear]` instead of
   inventing unreadable symbols;
9. stores the raw chronological transcription in `lectures.blackboard_latex`;
10. feeds that evidence into the ordinary course summarizer; and
11. appends the raw `### 黑板板书 LaTeX 转写` section verbatim to the final
    summary/email so it cannot be lost through summarization.

If board extraction or vision inference fails, the lecture gets
`error_stage=blackboard` and remains retryable. Already-processed functional
analysis lectures created by older versions are automatically queued once for
retrofit if `blackboard_latex` is missing.

## Why temporal selection is needed

A 2.5-hour recording sampled every 15 seconds contains about 600 raw frames.
The first real 泛函分析 run produced 588 candidate frames, so ordinary
perceptual-hash deduplication was ineffective: lecturer movement, camera noise,
and exposure changes made almost every frame look different.

The revised selector therefore uses a different rule. For each candidate frame
it measures low-resolution grayscale changes locally. A real writing/erasing
change often differs from the previous sample **and remains present in the next
sample**. A lecturer walking in front of the board is more likely to change in
both directions and is therefore less likely to be treated as a persistent
board event.

This heuristic is not the only source of coverage. Independently, the selector
keeps one stable representative frame from each one-minute window. This means a
missed change event cannot remove an entire part of the lecture timeline.

For very long lectures the 240-frame soft cap is applied after coverage anchors
are chosen. Anchors have priority; remaining slots are filled by the strongest
persistent-change events. The goal is to reduce API calls substantially without
simply changing to a coarse one-minute sampling interval that could miss
short-lived formulas.

## ModelScope configuration

`DASHSCOPE_API_KEY` is the historical environment-variable name used by this
project, but for the default ModelScope endpoint its value must be a valid
ModelScope access token.

The first real lecture run established that
`Qwen/Qwen3-VL-8B-Instruct` is callable through the configured ModelScope
serverless endpoint. `Qwen/Qwen3-VL-32B-Instruct` returned `has no provider
supported`, so it is no longer part of the default model list.

The vision client also tolerates partial batch output. If a batch contains four
frames but the model emits only three parseable frame wrappers, the three valid
results are kept and only the missing frame is retried individually.

Optional environment variables:

```text
BLACKBOARD_COURSES=泛函分析
BLACKBOARD_SAMPLE_SEC=15
BLACKBOARD_COVERAGE_SEC=60
BLACKBOARD_ANALYSIS_WIDTH=192
BLACKBOARD_CHANGE_THRESHOLD=24
BLACKBOARD_STABLE_THRESHOLD=14
BLACKBOARD_CHANGE_RATIO=0.008
BLACKBOARD_EVENT_GAP_SEC=30
BLACKBOARD_VISION_BATCH_SIZE=4
BLACKBOARD_VISION_MODELS=Qwen/Qwen3-VL-8B-Instruct
BLACKBOARD_VISION_MAX_EDGE=1600
BLACKBOARD_VISION_MAX_TOKENS=4096
BLACKBOARD_VISION_TIMEOUT=180
BLACKBOARD_MAX_FRAMES=240
BLACKBOARD_FFMPEG_TIMEOUT=1800
```

## Accuracy policy

The vision prompt is intentionally conservative. It must preserve visible
notation such as norms, dual spaces, weak and weak-star convergence, Greek and
calligraphic symbols, quantifiers, matrices, cases, arrows, and set operators.
It is forbidden from filling in a theorem or proof step from mathematical
context when the corresponding writing is not visible. Unreadable regions are
marked `[unclear]`.

The system therefore guarantees that functional-analysis lectures must produce
an inspectable board-transcription artifact before being marked complete; it
does not claim that every blurred or occluded chalk mark can be recovered with
100% visual accuracy.
