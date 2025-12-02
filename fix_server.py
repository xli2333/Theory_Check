
import os

file_path = 'server.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1-based line numbers to 0-based indices
# Keep lines 1 to 1534 -> indices 0 to 1533
part_a = lines[:1534]

# Keep lines 2704 to end -> indices 2703 to end
# Check if 2704 is indeed the main block start
# Line 2704 in file corresponds to index 2703
part_c = lines[2703:]

# Verify Part C starts with if __name__
if "if __name__" not in part_c[0]:
    print(f"Warning: Part C start '{part_c[0].strip()}' does not look like main block.")
    # Search for main block in the last 50 lines
    for i in range(len(lines) - 50, len(lines)):
        if "if __name__" in lines[i]:
            print(f"Found main block at line {i+1}")
            part_c = lines[i:]
            break

new_content = "".join(part_a) + "\n" + "".join(part_c)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Fixed {file_path}. New length: {len(new_content.splitlines())} lines.")
