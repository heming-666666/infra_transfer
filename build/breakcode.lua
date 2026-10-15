-- 行内代码在 / . _ : - , = 等字符后允许断行，避免 PDF 中长标识符溢出页边
local specials = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}",
  ["$"] = "\\$", ["&"] = "\\&", ["#"] = "\\#", ["%"] = "\\%",
  ["^"] = "\\textasciicircum{}", ["~"] = "\\textasciitilde{}",
  ["_"] = "\\_",
  -- Consolas/Segoe UI 缺这两个字形，改用有该字形的字体或数学符号
  ["▁"] = "{\\fontspec{NSimSun}▁}",
  ["∈"] = "\\ensuremath{\\in}",
}

-- 正文中的 ∈ 同样替换（正文字体 Segoe UI 缺字形）
function Str(el)
  if not FORMAT:match("latex") or not el.text:find("∈", 1, true) then return nil end
  local parts = {}
  for piece, sep in (el.text .. "∈"):gmatch("(.-)(∈)") do
    if piece ~= "" then parts[#parts + 1] = pandoc.Str(piece) end
    parts[#parts + 1] = pandoc.RawInline("latex", "\\ensuremath{\\in}")
  end
  table.remove(parts) -- 去掉为切分而追加的最后一个 ∈
  return parts
end
local breakers = { ["/"] = true, ["."] = true, ["_"] = true, [":"] = true,
  ["-"] = true, [","] = true, ["="] = true, ["("] = true, ["["] = true }

function Code(el)
  if not FORMAT:match("latex") then return nil end
  local out = {}
  for _, cp in utf8.codes(el.text) do
    local ch = utf8.char(cp)
    out[#out + 1] = specials[ch] or ch
    if breakers[ch] then out[#out + 1] = "\\allowbreak{}" end
  end
  return pandoc.RawInline("latex", "\\texttt{" .. table.concat(out) .. "}")
end
