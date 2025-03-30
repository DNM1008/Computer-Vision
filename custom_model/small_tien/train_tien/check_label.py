import os


def ensure_lines_start_with_zero(filename):
    with open(filename, "r+", encoding="utf-8") as file:
        lines = file.readlines()
        modified = False

        for i in range(len(lines)):
            stripped_line = lines[
                i
            ].lstrip()  # Remove leading spaces to check the first character

            if stripped_line and stripped_line[0] != "0":
                lines[i] = (
                    "0" + stripped_line[1:]
                )  # Replace the first character with "0"
                modified = True

        if modified:
            file.seek(0)
            file.writelines(lines)
            file.truncate()
            print(f"Fixed: {filename}")
        else:
            print(f"Already correct: {filename}")


for txt_file in os.listdir():
    if txt_file.endswith(".txt") and os.path.isfile(txt_file):
        ensure_lines_start_with_zero(txt_file)
