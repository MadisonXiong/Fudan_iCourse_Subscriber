"""Annotation policy and final quality gate for BlackboardEditor.

The base editor performs local de-duplication, global semantic consolidation,
faithfulness audit and normal LaTeX repair. This overlay allows concise
model-authored explanations, but requires them to be visibly marked as purple
``AI 补充`` blocks. After the base editor finishes, a strict local finalizer
checks the complete document for concrete renderer/transcription defects and
repairs only the affected small chunks against the raw vision evidence.

The expensive local LLM pass is resumable. Every successfully completed top-level
raw chunk is persisted to SQLite ``meta``. If a free inference quota is exhausted
halfway through a lecture, the next run starts at the first unfinished chunk
instead of paying for the completed chunks again. The checkpoint is bound to a
SHA-256 fingerprint of the exact raw vision transcription and current chunking
parameters, so stale progress can never be applied to different source material.
"""

from __future__ import annotations

import hashlib

from src.ai import blackboard_editor as _base
from src.ai.blackboard_finalizer import repair_final_anomalies
from src.data.blackboard_store import (
    clear_editor_checkpoint,
    load_editor_checkpoint,
    save_editor_checkpoint,
)


AI_NOTE_OPEN = (
    '<div data-ai-note="true" '
    'style="margin:12px 0;padding:10px 14px;'
    'border-left:4px solid #7c3aed;background:#f5f3ff;'
    'color:#5b21b6;border-radius:4px;">'
)
AI_NOTE_LABEL = '<strong style="color:#6d28d9;">AI 补充</strong><br>'
AI_NOTE_CLOSE = "</div>"
_EDITOR_CHECKPOINT_VERSION = 1


_base.GLOBAL_SYSTEM_PROMPT = rf"""
你是“整节数学板书总编辑”。输入已经经过第一轮局部去重，但不同分块之间仍可能出现相隔很远的语义重复，例如同一个 Minkowski 定理/证明、同一个 C([a,b]) 完备性证明、同一个 Hölder 回顾被再次完整写出。

请把整份第一轮稿整理成一份最终的、连续的、无重复的完整板书稿。

【A. 板书正文：必须忠实】
1. 识别同一数学内容的不同措辞、不同 LaTeX 写法、完整/不完整版本；若确认为同一内容，只保留一份最完整版本。
2. 如果后一次重复包含前一次没有的新证明步骤或条件，把这些新增信息并入第一次完整出现的位置；不要丢掉唯一内容。
3. 同一定理后面若课堂真的出现了新的证明、另一种证明、不同条件下的变体，则必须保留；只有实质相同的重复才删除。
4. 不做摘要，不缩写证明，不省略唯一公式或推导。
5. 板书正文不得混入你自己生成的数学知识、解释、评价或结论。课堂内容和 AI 内容必须视觉上严格分离。
6. 按课堂第一次出现的先后顺序组织。不要按教材知识体系重新排序。

【B. AI 补充：允许，但必须显式标记】
你可以在确实有助于理解的位置添加简短的解释、直觉、概念联系、符号说明或对证明关键一步的说明，但这些不是老师板书，必须全部放在下面这个紫色标记框中：

{AI_NOTE_OPEN}
{AI_NOTE_LABEL}
这里写 AI 补充内容。
{AI_NOTE_CLOSE}

必须严格遵守：
1. 任何输入中没有直接出现、由你补充的内容，都必须完整位于上述 `data-ai-note="true"` 容器中；绝不能伪装成普通板书正文。
2. 一条 AI 补充尽量简短，通常 1–4 句。不要为了“丰富笔记”到处加说明；只有确实能降低理解难度时才加。
3. AI 补充可以解释“这一步为什么这样做”“这个定义与前文有什么联系”“这里使用了什么标准思想”，但不要补写一套输入中不存在的新证明。
4. 如果补充中有数学公式，同样使用 `$...$` 或 `$$...$$`。
5. 不得修改上述容器的 `data-ai-note="true"` 标记和紫色 inline style；这样邮件中会明确显示为特殊颜色区域。
6. 不要自行生成普通正文形式的“补充说明”“附加讨论”。如果是你添加的内容，只能放进 AI 补充框。

【C. Markdown / LaTeX】
- 所有数学内容位于 `$...$` 或 `$$...$$`；
- 中文在数学环境外；
- 禁止 aligned、array、cases、gathered、split；
- 长推导拆成若干独立 display 公式；
- 不使用 Markdown 代码块；
- 保留 AI 补充框的原始 HTML，不要把它转义成代码；
- 绝不能输出 `##C##`、`xxxx`、`[unclear]`、乱码占位符。若来源确实无法判清，使用 `[转写存疑]`。

直接输出最终完整板书稿，不要解释你的处理过程。
""".strip()


_base.AUDIT_SYSTEM_PROMPT = r"""
你是“数学转写完整性与来源标记审计器”。你会看到：
A. 第一轮局部整理稿（它仍可能有重复，但应包含全部原始唯一板书内容）；
B. 全局整理后的候选最终稿。

只做审计，不要重写全文。检查三件事：
1. MISSING：A 中存在、B 中缺失的唯一板书内容，包括定义、条件、定理、例子、证明步骤、公式、反例、CHECK/Why。重复内容、等价措辞、纯格式差异不要报缺失。
2. UNMARKED_ADDITION：B 中出现、A 中没有，并且没有被完整包在 `<div data-ai-note="true" ...>...</div>` 中的新增解释、数学陈述、标题、评价或结论。
3. AI_NOTE_ERROR：AI 补充框存在，但边界损坏、嵌套混乱，或者其中的内容被混进普通板书正文。

注意：正确放在 `data-ai-note="true"` 紫色框中的 AI 解释是允许的，不属于错误，也不要要求删除。

如果三类问题都没有，只输出：
OK

否则严格输出：
ISSUES
MISSING:
- ...
UNMARKED_ADDITION:
- ...
AI_NOTE_ERROR:
- ...

没有某一类问题时写 `- none`。每一项尽量引用或紧贴原文，精确指出问题，不要自己补数学。
""".strip()


_base.CORRECTION_SYSTEM_PROMPT = rf"""
你是“数学板书最终校订器”。输入包含候选最终稿以及独立审计器发现的问题。

请只按审计报告修正：
- 把 MISSING 中的唯一板书内容恢复到课堂顺序中最合适的位置；
- 对 UNMARKED_ADDITION：若它是合理且有帮助的 AI 解释，不删除，而是完整移动进下面规定的 AI 补充框；若它明显不可靠或无价值，则删除；
- 修复 AI_NOTE_ERROR，使所有模型新增内容与老师板书严格分离；
- 同时继续消除明确的语义重复；
- 不得做摘要，不得删掉其他唯一证明步骤。

AI 补充必须使用且只能使用：
{AI_NOTE_OPEN}
{AI_NOTE_LABEL}
AI 补充内容
{AI_NOTE_CLOSE}

普通板书正文不得使用这个紫色框。保持 Markdown/LaTeX 可渲染：数学放入 `$...$` 或 `$$...$$`，不用 aligned/array/cases/gathered/split。
只输出修正后的完整正文。
""".strip()


_base.REPAIR_SYSTEM_PROMPT = r"""
你是 LaTeX 格式修复器。输入是一份已经完成内容整理的数学板书稿，其中可能包含 `<div data-ai-note="true" ...>...</div>` AI 补充框。

只能修复 Markdown/LaTeX 格式，不得删除、补充、总结、重排或改变任何数学内容。

要求：
- 所有数学内容必须在 `$...$` 或 `$$...$$` 内；
- 不得有裸露的 LaTeX 命令；
- 中文必须在数学环境外；
- 不使用 aligned/array/cases/gathered/split；
- 过长 display 公式拆成多个独立 `$$...$$`；
- 不使用代码块；
- 文本长度原则上应与输入接近；
- 必须原样保留所有 `data-ai-note="true"` 容器及其紫色 inline style，不得把 AI 补充变回普通正文。

直接输出修复后的全文。
""".strip()


class BlackboardEditor(_base.BlackboardEditor):
    """Annotated editor with durable local-pass resume and final preflight."""

    def __init__(self, *args, db=None, sub_id: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._checkpoint_db = db
        self._checkpoint_sub_id = str(sub_id) if sub_id is not None else ""

    @staticmethod
    def _source_fingerprint(raw_chunks: list[str]) -> str:
        digest = hashlib.sha256()
        for chunk in raw_chunks:
            digest.update(chunk.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _checkpoint_identity(self, raw_chunks: list[str]) -> dict:
        return {
            "version": _EDITOR_CHECKPOINT_VERSION,
            "source_sha256": self._source_fingerprint(raw_chunks),
            "chunk_count": len(raw_chunks),
            "raw_chunk_chars": self.raw_chunk_chars,
            "boundary_chars": self.local_boundary_chars,
        }

    def _load_local_progress(
        self, raw_chunks: list[str]
    ) -> tuple[list[str], list[str], list[float]]:
        if self._checkpoint_db is None or not self._checkpoint_sub_id:
            return [], [], []

        payload = load_editor_checkpoint(
            self._checkpoint_db, self._checkpoint_sub_id
        )
        if not payload:
            return [], [], []

        identity = self._checkpoint_identity(raw_chunks)
        if any(payload.get(key) != value for key, value in identity.items()):
            print(
                "[BlackboardEditor] stored editor checkpoint does not match "
                "the current raw source/chunking; ignoring stale progress",
                flush=True,
            )
            clear_editor_checkpoint(
                self._checkpoint_db, self._checkpoint_sub_id
            )
            return [], [], []

        outputs = payload.get("outputs")
        models = payload.get("models")
        ratios = payload.get("ratios")
        if not isinstance(outputs, list) or len(outputs) > len(raw_chunks):
            return [], [], []
        if not all(isinstance(item, str) and item.strip() for item in outputs):
            return [], [], []
        if not isinstance(models, list):
            models = []
        if not isinstance(ratios, list):
            ratios = []
        try:
            ratios = [float(value) for value in ratios]
        except (TypeError, ValueError):
            ratios = []

        if outputs:
            print(
                f"[BlackboardEditor] resumed durable local-pass checkpoint: "
                f"{len(outputs)}/{len(raw_chunks)} top-level chunk(s) already complete; "
                f"continuing at chunk {len(outputs) + 1}",
                flush=True,
            )
        return outputs, [str(x) for x in models], ratios

    def _save_local_progress(
        self,
        raw_chunks: list[str],
        outputs: list[str],
        models: list[str],
        ratios: list[float],
    ) -> None:
        if self._checkpoint_db is None or not self._checkpoint_sub_id:
            return
        payload = self._checkpoint_identity(raw_chunks)
        payload.update(
            {
                "stage": "local-pass",
                "outputs": outputs,
                "models": models,
                "ratios": ratios,
            }
        )
        save_editor_checkpoint(
            self._checkpoint_db,
            self._checkpoint_sub_id,
            payload,
        )
        print(
            f"[BlackboardEditor] checkpoint saved: "
            f"{len(outputs)}/{len(raw_chunks)} local chunk(s)",
            flush=True,
        )

    def _run_local_pass(self, raw_chunks: list[str]) -> tuple[str, list[str]]:
        outputs, models, ratios = self._load_local_progress(raw_chunks)
        previous_tail = outputs[-1] if outputs else ""
        completed = len(outputs)

        print(
            f"[BlackboardEditor] pass 1 / LLM local de-duplication: "
            f"{len(raw_chunks)} chunk(s), {completed} resumed",
            flush=True,
        )

        for index in range(completed, len(raw_chunks)):
            chunk = raw_chunks[index]
            text, used_models, chunk_ratios = self._edit_local_resilient(
                chunk,
                boundary=previous_tail[-self.local_boundary_chars:],
                previous_ratios=ratios,
                label=f"pass 1 chunk {index + 1}/{len(raw_chunks)}",
            )
            outputs.append(text)
            models.extend(used_models)
            ratios.extend(chunk_ratios)
            previous_tail = text

            # Save only after a complete top-level chunk passes all safety gates.
            # If a recursive .a succeeds but .b fails, the parent chunk is not
            # checkpointed and will be retried as one logical unit next time.
            self._save_local_progress(raw_chunks, outputs, models, ratios)

        return "\n\n".join(outputs).strip(), models

    def edit(self, raw_blackboard: str) -> tuple[str, str]:
        # Base ``edit`` calls our resumable _run_local_pass override.
        notes, model_label = super().edit(raw_blackboard)
        finalized = repair_final_anomalies(self, notes, raw_blackboard)

        all_models = [part for part in model_label.split("+") if part]
        all_models.extend(finalized.models)
        combined_model_label = "+".join(dict.fromkeys(all_models)) or "unknown"

        if finalized.repaired_chunks:
            print(
                f"[BlackboardEditor] final local preflight repaired "
                f"{finalized.repaired_chunks} chunk(s)",
                flush=True,
            )

        # Clear only after the entire annotated/finalized document succeeded.
        # Any quota/network failure before this point leaves the local progress
        # available for tomorrow's run.
        if self._checkpoint_db is not None and self._checkpoint_sub_id:
            clear_editor_checkpoint(
                self._checkpoint_db, self._checkpoint_sub_id
            )
            print(
                "[BlackboardEditor] complete document produced; "
                "local-pass checkpoint cleared",
                flush=True,
            )

        return finalized.text, combined_model_label


__all__ = ["BlackboardEditor", "AI_NOTE_OPEN", "AI_NOTE_LABEL", "AI_NOTE_CLOSE"]