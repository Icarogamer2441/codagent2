# Coda2 Agent

Coda2 is a command-line interface agent powered by the Google Gemini model. It is designed to assist with code development, file system management, and terminal operations.

## Features:

- **File System Interaction:** List, read, create, write, rename, and delete files and directories.
- **Terminal Execution:** Run arbitrary terminal commands.
- **Code Modification:** Modify files using search and replace or applying diffs.
- **Subthinking:** Plan actions in multiple steps before execution.
- **Creative Mode:** Enable more innovative suggestions and designs.
- **Automatic Code Verification:** Analyze modified files for potential issues.
- **Trust Mode:** Allow file operations without explicit confirmation.

## Tools Available:

- `list_directory`: Lists the contents of a directory.
- `read_file`: Reads the content of a file.
- `create_file`: Creates a new file.
- `write_file`: Overwrites an existing file.
- `rename_path`: Renames a file or directory.
- `delete_path`: Deletes a file or directory.
- `run_command`: Runs a terminal command.
- `modify_file`: Searches for and replaces text in a file.
- `apply_diff`: Applies a diff to a file.

## Getting Started:

setting up GOOGLE_API_KEY:
```bash
# linux:
$ export GOOGLE_API_KEY=your_api_key
# windows:
$ set GOOGLE_API_KEY=your_api_key
```

running the agent:
```bash
$ pip install -e .
$ coda2
```

now, just ask the agent to do something and it will do it!

## Special Commands:

- `/help`: Display help information.
- `/toggle-subthink`: Toggle subthinking mode.
- `/ststeps <number>`: Set the number of subthinking steps.
- `/creative-mode`: Toggle creative mode.
- `/analyze <file_path>`: Analyze a specific file for code issues.
- `/toggle-verify`: Toggle automatic code verification.
- `/toggle-trust`: Toggle trust mode.
- `quit`: Exit the agent.