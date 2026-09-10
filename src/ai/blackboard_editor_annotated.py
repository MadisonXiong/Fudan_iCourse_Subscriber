"""Policy overlay for BlackboardEditor with visually marked AI annotations.

The underlying editor keeps the raw-board preservation/de-duplication pipeline.
Only the *global* editing stage is allowed to add concise explanatory material,
and every such addition must be wrapped in a distinctive inline-styled HTML box
so the email cannot confuse model commentary with the professor's blackboard.

Raw/local transcription remains source-faithful.  This overlay intentionally
leaves LOCAL_SYSTEM_PROMPT unchanged and patches only the global/audit/correction
and format-repair policies before re-exporting BlackboardEditor.
"""

from __future__ import annotations

from src.ai import blackboard_editor as _base


AI_NOTE_OPEN = (
    '<div data-ai-note="true" '
    'style="margin:12px 0;padding:10px 14px;'
    'border-left:4px solid #7c3aed;background:#f5f3ff;'
    'color:#5b21b6;border-radius:4px;">'
)
AI_NOTE_LABEL = '<strong style="color:#6d28d9;">AI 补充</strong><br>'
AI_NOTE_CLOSE = "</div>"


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
- 禁止 aligned、array、cases；
- 长推导拆成若干独立 display 公式；
- 不使用 Markdown 代码块；
- 保留 AI 补充框的原始 HTML，不要把它转义成代码。

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

普通板书正文不得使用这个紫色框。保持 Markdown/LaTeX 可渲染：数学放入 `$...$` 或 `$$...$$`，不用 aligned/array/cases。
只输出修正后的完整正文。
""".strip()


_base.REPAIR_SYSTEM_PROMPT = r"""
你是 LaTeX 格式修复器。输入是一份已经完成内容整理的数学板书稿，其中可能包含 `<div data-ai-note="true" ...>...</div>` AI 补充框。

只能修复 Markdown/LaTeX 格式，不得删除、补充、总结、重排或改变任何数学内容。

要求：
- 所有数学内容必须在 `$...$` 或 `$$...$$` 内；
- 不得有裸露的 LaTeX 命令；
- 中文必须在数学环境外；
- 不使用 aligned/array/cases；
- 过长 display 公式拆成多个独立 `$$...$$`；
- 不使用代码块；
- 文本长度原则上应与输入接近；
- 必须原样保留所有 `data-ai-note="true"` 容器及其紫色 inline style，不得把 AI 补充变回普通正文。

直接输出修复后的全文。
""".strip()


# Re-export the existing implementation after installing the annotation policy.
BlackboardEditor = _base.BlackboardEditor

__all__ = ["BlackboardEditor", "AI_NOTE_OPEN", "AI_NOTE_LABEL", "AI_NOTE_CLOSE"]
