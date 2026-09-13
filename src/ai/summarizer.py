"""LLM-based course lecture summarization via ModelScope API.

Traceable summaries use source IDs supplied by ``bucketer.assemble_traceable``.
The model never writes literal timestamps: it cites Txx IDs, and this module
validates and deterministically renders those IDs as video ranges.
"""

from __future__ import annotations

import re
import time

from openai import OpenAI

from src.runtime import config


SYSTEM_PROMPT = r"""你是一个专业的课程助教。你的任务是根据用户提供的课程录音文本和ppt文字ocr部分，生成用于学生自学和期末复习的详细笔记。
1. **直接输出**：不要包含任何"好的"、"没问题"、"以下是总结"等客套话，不要输出全局课程名称大标题（由系统自动生成），直接开始总结即可。
2. **文本清洗**：语言必须通顺、逻辑清晰，严格去除口语化表达、重复句和无意义的录音识别错误等。内容可能被识别成同音字，需要通过学术语境修复。
3. **格式严格**：
   - 必须使用 Markdown 格式排版。
   - 标题结构清晰与级别限制：只允许使用三级及后续级别的标题（即只能使用`###`、`####`或`#####`），禁止使用 `#` 和 `##`。用清晰的标题组织结构。
   - 不得使用超过两级的缩进和超过两级嵌套的bullet point。
   - 尽可能用完整的段落来组织老师的讲解，适量使用bullet point，不得过度使用bullet point
   - 合理使用加粗、列表、表格、分级标题、段首小标题等形式来组织信息，确保结构清晰。
4. **公式规范**：所有数学公式或科学变量必须使用规范的 LaTeX 语法（行内公式用 $...$，行间公式用 $$...$$）。由于图床限制，latex公式中不要出现中文。
5. **忠于原文与详略得当**：总结必须有详有略，总体以详细风格为主，长度适宜或偏长（例如，对于90分钟长度的课程，总结长度应为5000字左右；135分钟的课程，应为8000字左右。输出和输入的长度压缩比例在1:8左右为宜），包含具体的推导细节、案例、文献或者核心概念，不要过度概括，也不要仅仅原文复述。禁止捏造录音中未提及的内容。
6. 你需要格外注意课程中是否提及了作业、考试、签到、组队等关键事项，如果有的话，用三级标题【课程事项提醒】标注在开头。
7. **文风示例**：以下是一个关于"梯度下降"的片段，展示了笔记总结过程中【错误的】和【正确的】的两种总结风格，请严格模仿后者。

【❌ 错误的风格】
## 梯度下降

**定义：**
- 梯度下降是一种优化算法
- 用于最小化损失函数
- 广泛应用于机器学习

【✅ 正确的风格】
### 梯度下降

梯度下降是最小化损失函数 $L(\theta)$ 的核心优化算法。其基本思想是沿着损失函数对参数 $\theta$ 的梯度的反方向迭代更新，每一步的更新公式为 $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$，其中 $\eta$ 称为学习率，控制每次更新的步长大小。

**学习率的选取至关重要**：若 $\eta$ 过大，参数更新幅度过猛，损失函数可能在最优点附近震荡甚至发散；若 $\eta$ 过小，收敛速度极慢，训练成本大幅上升。

8. **输入材料格式**：
   - 输入可能含带时间轴的【AI 校订语音转写】/【音频转录】以及【PPT 文字识别】。
   - PPT 文字可用于校正录音中的专业术语，但也可能有 OCR 错误。
   - 总结组织主线以讲师讲解为准，把 PPT 信息自然融合进去。
"""


TRACEABILITY_PROMPT = r"""

【视频溯源规则——必须严格执行】
输入被划分为真实视频来源块，例如：
`=== [T03] 视频 09:00–12:00 ===`

你不得直接生成或猜测任何时间。只能引用输入中真实存在的 Txx 来源 ID。

对每一个 `###`、`####`、`#####` 知识点标题，都必须在标题下一行单独输出一个来源标记：
`〔VIDEO:T03〕`
若该知识点跨连续多个来源块，输出：
`〔VIDEO:T03-T05〕`

要求：
1. 来源标记必须紧跟标题，不能放到段落末尾。
2. Txx 必须来自输入，禁止虚构。
3. 一个知识点只覆盖真正支撑它的最小连续视频范围，不要为了省事给整节课的大范围。
4. 如果一个知识点在两个不连续时段分别出现，分别写两个标记，例如：`〔VIDEO:T03〕〔VIDEO:T08〕`。
5. `【课程事项提醒】`若存在，也必须有视频来源标记。
6. 除这种 `〔VIDEO:...〕` 标记外，不要自己写“视频时间”“时间戳”等文字。系统会在模型输出后把来源 ID 转换成经过验证的实际视频时段。
"""


_TRACE_RE = re.compile(r"〔VIDEO:(T\d{2,3})(?:-(T\d{2,3}))?〕")
_HEADING_RE = re.compile(r"^(#{3,5})\s+(.+?)\s*$", re.MULTILINE)


def _fmt(sec: int) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _render_one_trace(
    match: re.Match,
    windows: dict[str, tuple[int, int]],
) -> str:
    first = match.group(1)
    last = match.group(2) or first
    if first not in windows or last not in windows:
        raise ValueError(f"summary cited unknown video source: {match.group(0)}")

    first_num = int(first[1:])
    last_num = int(last[1:])
    if last_num < first_num:
        raise ValueError(f"reversed video source range: {match.group(0)}")

    # Source IDs are ordinal and contiguous in the supplied prompt. Ensure the
    # model did not jump across missing IDs inside one claimed continuous span.
    ids = [f"T{i:02d}" for i in range(first_num, last_num + 1)]
    if any(source_id not in windows for source_id in ids):
        raise ValueError(f"non-contiguous video source range: {match.group(0)}")

    start = windows[first][0]
    end = windows[last][1]
    return f"**视频定位：{_fmt(start)}–{_fmt(end)}**"


def render_traceable_summary(
    text: str,
    windows: dict[str, tuple[int, int]],
) -> str:
    """Validate model source IDs and replace them with real video ranges."""
    if not windows:
        return text

    matches = list(_TRACE_RE.finditer(text))
    if not matches:
        raise ValueError("traceable summary contains no VIDEO source markers")

    rendered = _TRACE_RE.sub(lambda m: _render_one_trace(m, windows), text)

    # Every knowledge heading must have a source marker immediately after it.
    lines = rendered.splitlines()
    for index, line in enumerate(lines):
        if not re.match(r"^#{3,5}\s+\S", line.strip()):
            continue
        lookahead = "\n".join(lines[index + 1:index + 4])
        if "**视频定位：" not in lookahead:
            raise ValueError(f"heading lacks video provenance: {line.strip()}")

    # No internal Txx marker should survive rendering.
    if "〔VIDEO:" in rendered:
        raise ValueError("unparsed VIDEO source marker remains")
    return rendered


class Summarizer:
    """Course lecture summarizer with multi-provider fallback."""

    def __init__(self):
        self.providers = config.resolve_model_providers()
        if not self.providers:
            raise ValueError(
                "No model provider available. "
                "Set at least one provider's API key (e.g. DASHSCOPE_API_KEY)."
            )
        self._clients = {
            p["name"]: OpenAI(api_key=p["api_key"], base_url=p["base_url"])
            for p in self.providers
        }

    def _call_llm(
        self,
        client: OpenAI,
        model: str,
        title: str,
        content: str,
        *,
        traceable: bool = False,
    ) -> str:
        t0 = time.time()
        system_prompt = SYSTEM_PROMPT + (TRACEABILITY_PROMPT if traceable else "")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"以下是课程《{title}》的录音文本和课件证据，"
                        f"根据长度，你应该输出的字符数大约为{len(content)/7:.0f}字，"
                        f"请开始总结：\n\n{content}"
                    ),
                },
            ],
            timeout=180,
        )
        result = response.choices[0].message.content or ""
        elapsed = time.time() - t0
        usage = getattr(response, "usage", None)
        if usage is not None:
            print(
                f"[Summarizer] Done ({model}): "
                f"{len(content)} chars input → {len(result)} chars output"
                f" in {elapsed:.0f}s "
                f"(tokens: prompt={getattr(usage,'prompt_tokens','?')}, "
                f"completion={getattr(usage,'completion_tokens','?')})"
            )
        else:
            print(
                f"[Summarizer] Done ({model}): {len(content)} chars input"
                f" → {len(result)} chars output in {elapsed:.0f}s"
            )
        return result

    def summarize(
        self,
        title: str,
        content: str,
        *,
        video_windows: dict[str, tuple[int, int]] | None = None,
    ) -> tuple[str, str]:
        """Summarize lecture, optionally requiring validated video provenance.

        When ``video_windows`` is supplied, every model heading must cite only
        the source IDs present in that mapping. Unknown/fabricated IDs cause
        that model attempt to be rejected rather than displayed to the user.
        """
        if not content or not content.strip():
            return ("（内容为空）", "")

        errors = []
        traceable = bool(video_windows)
        for provider in self.providers:
            client = self._clients[provider["name"]]
            for model in provider["models"]:
                model_id = f"{provider['name']}/{model}"
                try:
                    result = self._call_llm(
                        client,
                        model,
                        title,
                        content,
                        traceable=traceable,
                    )
                    if video_windows:
                        result = render_traceable_summary(result, video_windows)
                    return (result, model_id)
                except Exception as e:
                    print(
                        f"[Summarizer] {model_id} failed/invalid: "
                        f"{type(e).__name__}: {e}"
                    )
                    errors.append(f"{model_id}: {e}")

        raise RuntimeError(
            "All LLM models failed or produced invalid provenance:\n"
            + "\n".join(errors)
        )
