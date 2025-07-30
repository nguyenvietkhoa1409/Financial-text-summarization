import os
import json

def clean_widgets_from_notebook(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        if "metadata" in nb and "widgets" in nb["metadata"]:
            print(f"Cleaning widgets from: {file_path}")
            del nb["metadata"]["widgets"]

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def clean_all_notebooks(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".ipynb"):
                full_path = os.path.join(dirpath, filename)
                clean_widgets_from_notebook(full_path)

if __name__ == "__main__":
    # Replace with your actual path or use "." for current directory
    clean_all_notebooks(".")