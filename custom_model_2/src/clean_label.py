"""
This program makes sure that the .txt files (label files) has 0 as its class
(start with 0)

"""

import os


def process_txt_files(directory):
    """
    Check every line in every .txt file in the given directory starts with 0, if
    not then replace the first char with 0

    Args:
        directory (str): directory of the .txt files
    """
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                if not line.startswith("0") and line.strip() != "":
                    line = "0" + line[1:] if len(line) > 1 else "0\n"
                new_lines.append(line)

            with open(filepath, "w") as file:
                file.writelines(new_lines)


if __name__ == "__main__":
    directory_path = "../data/source_data/small_tien"  # replace with your actual path
    process_txt_files(directory_path)
