-- Preserve FiCS source-provenance styling in LaTeX output.

local function has_class(classes, wanted)
  for _, value in ipairs(classes) do
    if value == wanted then
      return true
    end
  end
  return false
end

function Div(el)
  if has_class(el.classes, "ai-note") then
    local out = {pandoc.RawBlock("latex", "\\begin{ainote}")}
    for _, block in ipairs(el.content) do
      table.insert(out, block)
    end
    table.insert(out, pandoc.RawBlock("latex", "\\end{ainote}"))
    return out
  end

  if has_class(el.classes, "video-location") then
    local out = {pandoc.RawBlock("latex", "\\begin{videolocation}")}
    for _, block in ipairs(el.content) do
      table.insert(out, block)
    end
    table.insert(out, pandoc.RawBlock("latex", "\\end{videolocation}"))
    return out
  end

  if has_class(el.classes, "visual-restored-block") then
    local out = {pandoc.RawBlock("latex", "\\begin{visualrestoreblock}")}
    for _, block in ipairs(el.content) do
      table.insert(out, block)
    end
    table.insert(out, pandoc.RawBlock("latex", "\\end{visualrestoreblock}"))
    return out
  end

  return nil
end

function Span(el)
  if not has_class(el.classes, "visual-restored") then
    return nil
  end

  local out = {pandoc.RawInline("latex", "\\visualrestore{")}
  for _, inline in ipairs(el.content) do
    table.insert(out, inline)
  end
  table.insert(out, pandoc.RawInline("latex", "}"))
  return out
end
