import re

with open("Sections/2.Relatedwork.tex", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Section 3: shift [18]-[28] to [21]-[31]
for i in range(28, 17, -1):
    text = text.replace(f"[{i}]", f"[{i+3}]")

# 2. Section 2: shift [11]-[17] to [12]-[18]
for i in range(17, 10, -1):
    text = text.replace(f"[{i}]", f"[{i+1}]")

# 3. Add Pao to Section 1
add_pao_text = """將工業級設計規則檢查(DRC)整合至優化過程中。
[11]\textbf{2026年} $Q/V$ 的超緊湊奈米腔體設計，展現了在極小空間內極致的光學侷限能力。"""
text = text.replace("將工業級設計規則檢查(DRC)整合至優化過程中。", add_pao_text)

add_pao_table = """2021 & Hammond/Georgia Tech/MIT used differentiable transformations to embed foundry DRC rules, bridging theory and mass production.[10] \\\\[1.5em]

2026 & Pao utilized topology optimization to achieve ultra-compact, high $Q/V$ nanocavities, overcoming reliance on large-area periodic structures.[11] \\\\ % 💡 關鍵修正 1：這裡一定 \\\\ 結尾，底部的 \\hline 才會出現！"""
text = text.replace(
    "2021 & Hammond/Georgia Tech/MIT used differentiable transformations to embed foundry DRC rules, bridging theory and mass production.[10] \\\\ % 💡 關鍵修正 1：這裡一定要加 \\\\ 結尾，底部的 \\hline 才會出現！",
    add_pao_table
)

# 4. Add Hu and Chang to Section 2
add_hu_chang_text = """[18]\\textbf{西元2024年}Li等人的綜述確立了「代理模型結合退火(Surrogate --------的可靠性與高效性。
git push -u origin g等人進一步建立了一套結合機器學習與退火演算法的通用設計框架，並應用於超振盪透鏡（SOL）的反向設計，成功打破了小焦點與高旁瓣的權衡難題，為量子相容的光學元件設計鋪平了道路。"""
text = text.replace(
    "[18]\\textbf{西元2024年}Zhu與Li等人的綜述確立了「代理模型結合退火(Surrogate Model + Annealing)」混合架構在複雜物理反向設計中的可靠性與高效性。",
    add_hu_chang_text
)

add_hu_chang_table = """2024 & Zhu/Chemical Reviews published a comprehensive review endorsing the ``surrogate model + annealing'' paradigm for inverse design.[18] \\\\[1.5em]

2025 & Hu applied the FMQA framework to optical filters and metastructures, resolving conflicts between long-range interference and quadratic coupling.[19] \\\\[1.5em]

2026 & Chang established a universal ML-annealing framework for superoscillatory lenses, overcoming the small-focus and high-sidelobe trade-off.[20] \\\\"""
text = text.replace(
    "2024 & Zhu/Chemical Reviews published a comprehensive review endorsing the ``surrogate model + annealing'' paradigm for inverse design.[18] \\\\",
    add_hu_chang_table
)

with open("Sections/2.Relatedwork.tex", "w", encoding="utf-8") as f:
    f.write(text)

print("Done")
