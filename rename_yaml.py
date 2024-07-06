import os

def rename_files_recursively(folder_path, old_name, new_name):
    # Traverse all files and directories in the folder recursively
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Check if the file name contains the old name
            if old_name in file_name:
                # Get the full path of the file
                file_path = os.path.join(root, file_name)
                # Rename the file
                new_file_name = file_name.replace(old_name, new_name)
                new_file_path = os.path.join(root, new_file_name)
                os.rename(file_path, new_file_path)
                print(f"Renamed: {file_path} -> {new_file_path}")

# Example usage
folder_path = './config/selfrag_reproduction/'
old_name = 'selfrag_llama3_8b'
new_name = 'selfrag_llama3_8B'

rename_files_recursively(folder_path, old_name, new_name)