import os


def scan_files(path: str):
    for entry in os.scandir(path):
        if entry.is_file():
            yield entry.path
