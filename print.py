import os

# Function to process files in a directory
def process_files(directory):
    # Get the list of files and subdirectories in the current directory
    entries = os.listdir(directory)

    # Iterate over each entry
    for entry in entries:
        # Get the full path of the entry
        entry_path = os.path.join(directory, entry)

        # Check if the entry is a file
        if os.path.isfile(entry_path):
            # Check if the file ends with '.py'
            if entry_path.endswith('.py'):
                # Open the file and read its contents
                with open(entry_path, 'r') as f:
                    contents = f.read()

                # Print the filename and contents with markdown-like backticks
                print(f"```{entry_path}")
                print(contents)
                print("```")
        # Check if the entry is a directory
        elif os.path.isdir(entry_path):
            # Recursively process the subdirectory
            process_files(entry_path)

# Start processing files from the current directory
process_files('.')
