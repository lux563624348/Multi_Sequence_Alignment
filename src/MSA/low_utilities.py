from pathlib import Path

def Create_File_with_Parent_Dirs(file_path):
    # Create a Path object
    path = Path(file_path)

    # Create parent directories if they do not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create the file (this will also create the file if it doesn't exist)
    path.touch(exist_ok=True)

    print(f"File {file_path} created successfully with parent directories.")
    return None

def Delete_File(file_path):
    # Create a Path object
    path = Path(file_path)

    # Check if the file exists
    if path.is_file():
        # Delete the file
        path.unlink()
        print(f"File {file_path} deleted successfully.")
    else:
        print(f"File {file_path} does not exist.")
    return None