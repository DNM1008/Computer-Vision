import os


def ensure_first_char_zero(filename):
    with open(filename, "r+", encoding="utf-8") as file:
        lines = file.readlines()
        file.seek(0)
        for i, line in enumerate(lines):
            if line and line[0] != "0":
                lines[i] = "0" + line[1:]
        file.writelines(lines)
        file.truncate()
        print(f"Fixed: {filename}")


for txt_file in os.listdir():
    if txt_file.endswith(".txt") and os.path.isfile(txt_file):
        ensure_first_char_zero(txt_file)
