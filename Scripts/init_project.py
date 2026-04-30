import os
from pathlib import Path

# =========================
# Project configuration
# =========================

PROJECT_NAME = "my_data_science_project"

BASE_DIR = Path(__file__).resolve().parent.parent

FOLDERS = [
    "data/raw",
    "data/processed",
    "data/external",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/visualization",
    "app",
    "models",
    "reports/figures",
    "tests",
    "scripts"
]

FILES = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/models/__init__.py",
    "src/visualization/__init__.py",
    "config.py",
    "requirements.txt",
    ".gitignore",
    "README.md"
]


# =========================
# CREATE PROJECT STRUCTURE
# =========================

def create_folders():
    for folder in FOLDERS:
        path = BASE_DIR / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"✔ Created folder: {path}")


def create_files():
    for file in FILES:
        path = BASE_DIR / file
        if not path.exists():
            path.write_text("")
            print(f"✔ Created file: {path}")


def main():
    print(f"\nInitializing project: {PROJECT_NAME}\n")
    create_folders()
    create_files()
    print("\n Project structure created successfully!\n")


if __name__ == "__main__":
    main()