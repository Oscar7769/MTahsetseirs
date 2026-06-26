import re

with open("Sections/2.Relatedwork.tex", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Replace the specific text [15] with [6] and Table 2 citation [15] with [5, 6]
text = text.replace("[15]\\textbf{西元2015年}Shen與Menon", "[6]\\textbf{西元2015年}Shen與Menon")
text = text.replace("pixelated design school.[15]", "pixelated design school.[5, 6]")

# 2. Shift all [16]-[31] down by 1
for i in range(16, 32):
    text = text.replace(f"[{i}]", f"[{i-1}]")

with open("Sections/2.Relatedwork.tex", "w", encoding="utf-8") as f:
    f.write(text)

with open("Sections/appendix.tex", "r", encoding="utf-8") as f:
    app_text = f.read()

# 1. Remove item [15] from appendix
lines = app_text.split('\n')
new_lines = []
skip_next = False
for line in lines:
    if "\\item[{[15]}] Shen, B., Wang" in line:
        continue # Skip this line
    new_lines.append(line)

app_text = '\n'.join(new_lines)

# 2. Shift all [16]-[31] down by 1 in appendix
for i in range(16, 32):
    app_text = app_text.replace(f"\\item[{{[{i}]}}]", f"\\item[{{[{i-1}]}}]")

with open("Sections/appendix.tex", "w", encoding="utf-8") as f:
    f.write(app_text)

print("Done fixing duplicates")
