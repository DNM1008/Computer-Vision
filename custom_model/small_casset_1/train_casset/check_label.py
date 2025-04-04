import os


def ensure_first_char_zero(filename):
    with open(filename, "r+", encoding="utf-8") as file:
        content = file.read()
        if content and content[0] != "0":
            file.seek(0)
            file.write("0" + content[1:])
            file.truncate()
            print(f"Fixed: {filename}")
        else:
            print(f"Already correct: {filename}")


for txt_file in os.listdir():
    if txt_file.endswith(".txt") and os.path.isfile(txt_file):
        ensure_first_char_zero(txt_file)
