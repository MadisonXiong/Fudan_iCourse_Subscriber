# Blackboard LaTeX transcription

This fork adds a required handwritten-board transcription path for courses
whose title matches `BLACKBOARD_COURSES`.  The default whitelist contains only
`泛函分析`; every other course continues to use the original ASR + PPT OCR +
summary pipeline and does not spend vision-model calls.

## Functional-analysis behavior

For a matching course, a lecture is not considered complete until the real
lecture video has produced a non-empty blackboard transcription.  The runtime:

1. obtains the same signed iCourse MP4 used by the audio pipeline;
2. samples one video frame every 15 seconds by default;
3. removes only near-identical consecutive frames with perceptual hashing;
4. sends chronological frame batches to ModelScope Qwen3-VL;
5. classifies projected slides/screens as `no_board` and transcribes genuine
   handwritten blackboard/whiteboard frames;
6. requires math to be emitted as LaTeX and uses `[unclear]` instead of
   inventing unreadable symbols;
7. stores the raw chronological transcription in `lectures.blackboard_latex`;
8. feeds that evidence into the ordinary course summarizer; and
9. appends the raw `### 黑板板书 LaTeX 转写` section verbatim to the final
   summary/email so it cannot be lost through summarization.

If board extraction or vision inference fails, the lecture gets
`error_stage=blackboard` and remains retryable.  Already-processed functional
analysis lectures created by older versions are automatically queued once for
retrofit if `blackboard_latex` is missing.

## ModelScope configuration

`DASHSCOPE_API_KEY` is the historical environment-variable name used by this
project, but for the default ModelScope endpoint its value must be a valid
ModelScope access token.  The default board models are:

```text
Qwen/Qwen3-VL-32B-Instruct
Qwen/Qwen3-VL-8B-Instruct
```

The 32B model is tried first for mathematical handwriting quality; the 8B
model is a fallback.  Both use ModelScope's OpenAI-compatible API.

Optional environment variables:

```text
BLACKBOARD_COURSES=泛函分析
BLACKBOARD_SAMPLE_SEC=15
BLACKBOARD_HASH_DISTANCE=2
BLACKBOARD_VISION_BATCH_SIZE=6
BLACKBOARD_VISION_MODELS=Qwen/Qwen3-VL-32B-Instruct,Qwen/Qwen3-VL-8B-Instruct
BLACKBOARD_VISION_MAX_EDGE=1600
BLACKBOARD_VISION_TIMEOUT=180
BLACKBOARD_MAX_FRAMES=0
BLACKBOARD_FFMPEG_TIMEOUT=1800
```

`BLACKBOARD_MAX_FRAMES=0` means no cap.  This is the default because the goal
for functional analysis is completeness rather than minimizing API calls.

## Accuracy policy

The vision prompt is intentionally conservative.  It must preserve visible
notation such as norms, dual spaces, weak and weak-star convergence, Greek and
calligraphic symbols, quantifiers, matrices, cases, arrows, and set operators.
It is forbidden from filling in a theorem or proof step from mathematical
context when the corresponding writing is not visible.  Unreadable regions
are marked `[unclear]`.

The system therefore guarantees that functional-analysis lectures must produce
an inspectable board-transcription artifact before being marked complete; it
does not claim that every blurred or occluded chalk mark can be recovered with
100% visual accuracy.
