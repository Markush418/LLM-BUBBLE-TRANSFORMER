filepath = r"C:\Users\negocio\Desktop\LLM-BUBBLE\models\qwen3_gqa_bubble_wrapper.py"

with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(125, 160):
    line = lines[i]
    visual = line.rstrip('\n').replace('\t', '[TAB]').replace(' ', '.')
    print(f"{i+1}: {visual[:100]}")
