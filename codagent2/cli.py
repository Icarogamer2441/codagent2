import os
import sys
import subprocess
import shutil  # Import shutil for directory operations
import time
import json  # Import for serializing/deserializing chat history summaries
from datetime import datetime
# Removed dotenv import as key is from user env vars
# from dotenv import load_dotenv
# Removed the patch library as we're using search/replace instead
# import patch
import re
# Add tokenize for code analysis
import tokenize
import io
import difflib
# Import tempfile for temporary file operations
import tempfile
# Import patch for applying diffs
import unidiff

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
# Removed Live import as we are removing streaming
# from rich.live import Live
# Re-import Rich Prompt components
from rich.prompt import Prompt, Confirm
from rich.status import Status
from rich.syntax import Syntax
from rich.errors import MarkupError  # Import MarkupError
# from rich.text import Text # Already imported
from rich.tree import Tree  # Import Tree for directory listing
# Import escape to handle special characters in Rich markup
from rich.markup import escape
from rich.table import Table

# Re-add prompt_toolkit for arrow keys support
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

# Import LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# Removed Runnable import
# from langchain_core.runnables import Runnable # Import Runnable for streaming

# Removed load_dotenv() - GOOGLE_API_KEY is expected in the user's environment
# load_dotenv()

# Initialize Rich Console
console = Console()

# Shared thinking_status that will be accessible globally
thinking_status = None

# Subthinking configuration
subthinking_enabled = False
subthinking_steps = 3

# Creative mode configuration
creative_mode = False

# Trust mode configuration
trust_mode_enabled = False

# Auto verification configuration
auto_verify_enabled = True

# Context tracking for recent actions and modified files
agent_context = {
    "last_modified_files": [],
    "last_read_files": [],
    "last_actions": [],
    "reported_errors": []
}

# Path to history file
HISTORY_FILE_PATH = '.coda2.history.json'

# --- Context/History Management Functions ---

def save_history_to_file():
    """Saves the current agent_context history to a JSON file."""
    try:
        # Prepare the history data
        history_data = {
            "timestamp": datetime.now().isoformat(),
            "context": agent_context
        }
        
        # Load existing history if it exists
        existing_history = []
        if os.path.exists(HISTORY_FILE_PATH):
            try:
                with open(HISTORY_FILE_PATH, 'r', encoding='utf-8') as f:
                    existing_history = json.load(f)
                    if not isinstance(existing_history, list):
                        existing_history = []
            except json.JSONDecodeError:
                # If file is corrupted, start with empty history
                existing_history = []
        
        # Add new history and keep only the last 50 entries
        existing_history.insert(0, history_data)
        if len(existing_history) > 50:
            existing_history = existing_history[:50]
        
        # Write to file
        with open(HISTORY_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing_history, f, indent=2)
            
    except Exception as e:
        console.print(f"[bold yellow]Warning: Failed to save history: {e}[/bold yellow]")

def load_history_from_file():
    """Loads the history from the JSON file and updates agent_context."""
    global agent_context
    
    if not os.path.exists(HISTORY_FILE_PATH):
        return
    
    try:
        with open(HISTORY_FILE_PATH, 'r', encoding='utf-8') as f:
            history_entries = json.load(f)
            
            if history_entries and isinstance(history_entries, list) and history_entries[0].get('context'):
                # Load the most recent context
                stored_context = history_entries[0]['context']
                
                # Update agent_context with stored values, preserving structure
                for key in agent_context:
                    if key in stored_context and stored_context[key]:
                        agent_context[key] = stored_context[key]
                        
                console.print("[bold blue]Loaded previous session context.[/bold blue]")
    except Exception as e:
        console.print(f"[bold yellow]Warning: Failed to load history: {e}[/bold yellow]")

# --- Tool Definitions ---

# Helper for custom confirmation prompt using prompt_toolkit


def ask_confirmation(prompt_text: str) -> bool:
    """Asks the user for y/n confirmation using prompt_toolkit."""
    global thinking_status, trust_mode_enabled
    
    # If trust mode is enabled for file operations, return True automatically
    # But only if the prompt suggests this is a file operation (not terminal command)
    if trust_mode_enabled and any(file_term in prompt_text.lower() for file_term in 
                                 ["file", "directory", "folder", "path", "create", "write", 
                                  "overwrite", "modify", "rename", "applying"]):
        console.print(f"[dim]Trust mode enabled: Auto-confirming file operation.[/dim]")
        console.print("[bold green]Auto-confirmed.[/bold green]")
        return True
    
    session = PromptSession()
    while True:
        try:
            # Add extra spacing before the prompt to make it more visible
            console.print("")
            # Important: Turn off the thinking status if it's active
            # This ensures it doesn't overlap with user input
            if thinking_status:
                thinking_status.stop()

            response = session.prompt(
                HTML(f"<yellow><b>>> {prompt_text}</b></yellow> (y/n) ")).strip().lower()
            console.print("")  # Add spacing after input
            if response == 'y':
                console.print("[bold green]Confirmed.[/bold green]")
                return True
            elif response == 'n':
                console.print("[bold red]Cancelled.[/bold red]")
                return False
            else:
                console.print(
                    "[bold red]Invalid input.[/bold red] Please type 'y' for yes or 'n' for no.")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold red]Input cancelled.[/bold red]")
            return False

# --- Context/History Management Functions ---


def summarize_conversation(messages, max_summary_length=500):
    """Summarizes a conversation to reduce context size."""
    # Extract the actual content from messages
    conversation_text = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Agent'}: {msg.content}"
                                  for msg in messages])

    # For this initial implementation, we'll do a simple truncation summary
    # In a more robust implementation, you might use an LLM to generate a true summary
    if len(conversation_text) > max_summary_length:
        return f"Previous conversation summary (truncated): {conversation_text[:max_summary_length]}..."
    return conversation_text


def summarize_chat_history(chat_history, keep_last_n):
    """Keeps the most recent messages and summarizes older ones to maintain context."""
    if len(chat_history) <= keep_last_n:
        return chat_history
    
    # Extract older messages to summarize
    older_messages = chat_history[:-keep_last_n]
    recent_messages = chat_history[-keep_last_n:]
    
    # Create a summary of older messages
    summary_text = summarize_conversation(older_messages)
    summary_message = SystemMessage(content=f"Previous conversation summary: {summary_text}")
    
    # Return the summary as a system message plus the recent messages
    return [summary_message] + recent_messages


def manage_chat_history(chat_history, max_messages=10, max_tokens=4000):
    """Manages chat history to prevent context explosion.

    Args:
        chat_history: List of message objects
        max_messages: Maximum number of recent messages to keep in full
        max_tokens: Approximate token budget for history (rough estimate)

    Returns:
        A pruned chat history with summaries replacing older messages
    """
    if len(chat_history) <= max_messages:
        return chat_history

    # Keep the most recent messages
    recent_messages = chat_history[-max_messages:]
    older_messages = chat_history[:-max_messages]

    # Create a summary of older messages
    summary = summarize_conversation(older_messages)

    # Replace older messages with a summary system message
    return [SystemMessage(content=f"CONVERSATION HISTORY: {summary}")] + recent_messages


def save_chat_history(chat_history, filepath):
    """Saves chat history to a file."""
    # Convert message objects to serializable dict
    serializable_history = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            serializable_history.append(
                {"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serializable_history.append({"role": "ai", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            serializable_history.append(
                {"role": "system", "content": msg.content})

    try:
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f)
        return True
    except Exception as e:
        console.print(f"[bold red]Error saving chat history: {e}[/bold red]")
        return False


def load_chat_history(filepath):
    """Loads chat history from a file."""
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r') as f:
            serialized_history = json.load(f)

        # Convert serialized history back to message objects
        chat_history = []
        for item in serialized_history:
            if item["role"] == "human":
                chat_history.append(HumanMessage(content=item["content"]))
            elif item["role"] == "ai":
                chat_history.append(AIMessage(content=item["content"]))
            elif item["role"] == "system":
                chat_history.append(SystemMessage(content=item["content"]))

        return chat_history
    except Exception as e:
        console.print(f"[bold red]Error loading chat history: {e}[/bold red]")
        return []


@tool
def list_directory(path: str = '.') -> str:
    """Lists the contents of a directory. Defaults to the current directory if no path is provided."""
    # Fix linter warning and improve path validation flow
    if not os.path.exists(path):
        console.print(Panel(
            f"[bold red]Error:[/bold red] Path '{escape(path)}' not found.", expand=False))
        return f"Error: Path '{path}' not found."
    if not os.path.isdir(path):
        console.print(Panel(
            f"[bold red]Error:[/bold red] Path '{escape(path)}' is not a directory.", expand=False))
        return f"Error: Path '{path}' is not a directory."

    console.print(Panel(
        f"[bold blue]Listing contents of:[/bold blue] {escape(path)}", expand=False))

    # FIX: Rewrite the tree creation logic to properly handle Rich markup
    try:
        abs_path = os.path.abspath(path)
        # Use the folder name as the root label, properly escaped
        root_label = escape(path)
        # Create tree without link markup that was causing errors
        tree = Tree(f":open_file_folder: {root_label}")

        contents = os.listdir(path)
        if not contents:
            tree.add("[italic]Empty directory[/italic]")
        else:
            # Sort contents for consistent output
            contents.sort()
            for item in contents:
                item_path = os.path.join(path, item)
                # Check if the path exists before adding to tree
                if os.path.exists(item_path):
                    # Use simple escaped labels without link markup
                    if os.path.isdir(item_path):
                        tree.add(f":file_folder: {escape(item)}/")
                    else:
                        tree.add(f":page_facing_up: {escape(item)}")
                else:
                    tree.add(
                        f"[italic red]Deleted during listing: {escape(item)}[/italic red]")

        # This should not throw a MarkupError now
        console.print(tree)
        # Return a simplified string representation for the agent
        return f"Successfully listed contents of {path}:\n{', '.join(contents)}"
    except PermissionError:
        console.print(Panel(
            f"[bold red]Error:[/bold red] Permission denied to list directory '{escape(path)}'.", expand=False))
        return f"Error: Permission denied to list directory '{path}'."
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/bold red] An unexpected error occurred while listing '{escape(path)}'. Reason: {e}", expand=False))
        return f"Error: An unexpected error occurred while listing '{path}'. Reason: {e}"


@tool
def read_file(path: str) -> str:
    """Reads the content of a file."""
    global agent_context
    
    try:
        # Ensure the path exists and is a file
        if not os.path.exists(path):
            console.print(Panel(
                f"[bold red]Error:[/bold red] File '{escape(path)}' not found.", expand=False))
            return f"Error: File '{path}' not found."
        if not os.path.isfile(path):
            console.print(Panel(
                f"[bold red]Error:[/bold red] Path '{escape(path)}' is not a file.", expand=False))
            return f"Error: Path '{path}' is not a file."

        console.print(
            Panel(f"[bold blue]Reading file:[/bold blue] {escape(path)}", expand=False))
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        console.print(Panel(Syntax(content, "text", theme="monokai", line_numbers=True),
                      # Escape path in title
                            title=f"[bold blue]Content of {escape(path)}[/bold blue]", expand=False))
        
        # Track this file read operation
        if path not in agent_context["last_read_files"]:
            agent_context["last_read_files"].insert(0, path)
            # Keep the list to a reasonable size
            if len(agent_context["last_read_files"]) > 5:
                agent_context["last_read_files"].pop()
        
        # Add this action to context
        agent_context["last_actions"].insert(0, {"action": "read_file", "path": path})
        if len(agent_context["last_actions"]) > 10:
            agent_context["last_actions"].pop()
            
        return content  # Return the actual content to the agent
    except FileNotFoundError:  # This is technically redundant due to the check above, but good practice
        console.print(Panel(
            f"[bold red]Error:[/bold red] File '{escape(path)}' not found.", expand=False))
        return f"Error: File '{path}' not found."
    except PermissionError:
        console.print(Panel(
            f"[bold red]Error:[/bold red] Permission denied to read file '{escape(path)}'.", expand=False))
        return f"Error: Permission denied to read file '{path}'."
    except UnicodeDecodeError:
        console.print(Panel(
            f"[bold red]Error:[/bold red] Could not decode file '{escape(path)}' with utf-8. It might be a binary file.", expand=False))
        return f"Error: Could not decode file '{path}' with utf-8. It might be a binary file."
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/bold red] An unexpected error occurred while reading '{escape(path)}'. Reason: {e}", expand=False))
        return f"Error: An unexpected error occurred while reading '{path}'. Reason: {e}"


def analyze_code(file_path, content):
    """Analyzes code for duplications and potential errors.
    
    Args:
        file_path: Path to the file being edited
        content: The content to analyze
        
    Returns:
        A tuple of (has_issues, analysis_result) where analysis_result contains
        the detected issues and suggested fixes.
    """
    global agent_context, thinking_status, creative_mode
    
    # Don't analyze non-code files
    file_ext = os.path.splitext(file_path)[1].lower()
    code_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php', '.html', '.css', '.scss']
    if file_ext not in code_extensions:
        return False, "Not a code file, skipping analysis."
    
    # Prepare for analysis
    console.print(Panel("[bold blue]Analyzing code for issues...[/bold blue]", expand=False))
    
    try:
        # Make sure thinking status is stopped before asking for user input
        if thinking_status:
            thinking_status.stop()
        
        # Get Gemini to analyze the code
        try:
            # Configure LLM for analysis
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-preview-04-17", 
                google_api_key=os.environ.get("GOOGLE_API_KEY"), 
                temperature=0  # Use zero temperature for analysis
            )
            
            # Set up thinking_status
            thinking_status = Status("Analyzing code...", spinner="dots", console=console)
            thinking_status.start()
            
            # Split longer files into sections if needed
            sections = []
            lines = content.split('\n')
            
            if len(lines) > 1000:
                # For very large files, split into 1000-line chunks
                for i in range(0, len(lines), 1000):
                    sections.append('\n'.join(lines[i:i+1000]))
            else:
                sections = [content]
                
            # Analyze each section
            all_issues = []
            
            for i, section in enumerate(sections):
                if thinking_status:
                    thinking_status.update(f"Analyzing code section {i+1}/{len(sections)}...")
                
                # Prepare the analysis prompt
                analysis_prompt = f"""You are a code review expert. Analyze the following code from {file_path} for:
1. Duplicated code blocks or patterns
2. Syntax errors
3. Logic errors or bugs
4. Style issues or anti-patterns
5. Security vulnerabilities

For each issue, provide:
1. A clear description of the problem
2. The line numbers or code region affected
3. A suggested fix

Only report actual issues - don't invent problems if none exist. If there are no issues, respond with "NO_ISSUES_FOUND".

Here's the code:

```
{section}
```
"""
                # Run the analysis
                messages = [
                    SystemMessage(content="You're a code analysis expert. Only report genuine issues with specific fixes."),
                    HumanMessage(content=analysis_prompt)
                ]
                
                try:
                    response = llm.invoke(messages)
                    analysis = response.content.strip()
                    
                    # Check if issues were found
                    if "NO_ISSUES_FOUND" not in analysis:
                        all_issues.append(analysis)
                    
                except Exception as e:
                    console.print(f"[bold yellow]Warning: Error during code analysis: {e}[/bold yellow]")
                    all_issues.append(f"Analysis error: {str(e)}")
            
            # Stop thinking status
            if thinking_status:
                thinking_status.stop()
                thinking_status = None
            
            # Combine and process results
            if all_issues:
                issues_text = "\n\n".join(all_issues)
                console.print(Panel(issues_text, title="[bold yellow]Code Issues Detected[/bold yellow]", expand=False))
                
                # If issues were found, ask if we should attempt fixes
                if ask_confirmation("Would you like the agent to fix these issues? (y/n)"):
                    # Use Gemini to apply fixes
                    thinking_status = Status("Generating fixes...", spinner="dots", console=console)
                    thinking_status.start()
                    
                    fix_prompt = f"""Based on the detected issues in the code, please provide the corrected code for {file_path}.
Issues detected:
{issues_text}

Original code:
```
{content}
```

Provide ONLY the corrected code without explanations or markers. Make minimal necessary changes to fix the issues while preserving the original functionality and intent.
"""
                    
                    messages = [
                        SystemMessage(content="You're a code repair expert. Generate fixed code with minimal changes."),
                        HumanMessage(content=fix_prompt)
                    ]
                    
                    try:
                        response = llm.invoke(messages)
                        fixed_code = response.content.strip()
                        
                        # Extract code block if there's markdown formatting
                        if "```" in fixed_code:
                            # Split by ``` to get the code block
                            code_parts = fixed_code.split("```")
                            if len(code_parts) >= 3:  # Proper markdown code block
                                # Get the middle part (the actual code)
                                fixed_code = code_parts[1]
                                # Remove language identifier if present
                                if fixed_code and "\n" in fixed_code:
                                    # First line might contain the language identifier
                                    first_line, rest = fixed_code.split("\n", 1)
                                    # If first line doesn't contain actual code (just language id)
                                    if first_line.strip() and not any(c in first_line for c in "();={}/\\"):
                                        fixed_code = rest
                            else:
                                # Fallback to original approach if structure is unexpected
                                fixed_code = fixed_code.split("```")[1]
                        
                            fixed_code = fixed_code.strip()
                        
                        # Stop thinking status
                        if thinking_status:
                            thinking_status.stop()
                            thinking_status = None
                        
                        # Show diff of changes
                        diff = difflib.unified_diff(
                            content.splitlines(),
                            fixed_code.splitlines(),
                            lineterm='',
                            n=3
                        )
                        diff_text = '\n'.join(diff)
                        
                        if diff_text.strip():
                            console.print(Panel(Syntax(diff_text, "diff", theme="monokai"),
                                          title="[bold blue]Proposed Fixes[/bold blue]", expand=False))
                            
                            # Ask for confirmation to apply fixes
                            if ask_confirmation(f"Apply these fixes to {file_path}? (y/n)"):
                                # Apply fixes
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(fixed_code)
                                console.print(Panel(
                                    f"[bold green]Success:[/bold green] Applied fixes to '{escape(file_path)}'.", expand=False))
                                return True, "Issues fixed."
                            else:
                                console.print(Panel(
                                    "[bold yellow]Fixes not applied.[/bold yellow]", expand=False))
                                return True, "Issues detected but fixes not applied."
                        else:
                            console.print(Panel(
                                "[bold yellow]No significant changes in the fixed code.[/bold yellow]", expand=False))
                            return False, "No significant fixes needed."
                    
                    except Exception as e:
                        if thinking_status:
                            thinking_status.stop()
                            thinking_status = None
                        console.print(f"[bold red]Error generating fixes: {e}[/bold red]")
                        return True, f"Issues detected but could not generate fixes: {e}"
                
                return True, "Issues detected but not fixed."
            else:
                console.print(Panel("[bold green]No issues detected in the code.[/bold green]", expand=False))
                return False, "No issues detected."
                
        except Exception as e:
            console.print(f"[bold red]Error during code analysis: {e}[/bold red]")
            return False, f"Error during analysis: {e}"
            
    finally:
        # Ensure thinking status is stopped
        if thinking_status:
            thinking_status.stop()
            thinking_status = None
    
    return False, "Analysis completed."

@tool
def create_file(path: str, content: str) -> str:
    """Creates a new file at the specified path with the given content. Requires user confirmation."""
    # Get the global thinking_status if it exists
    global thinking_status, agent_context

    # Process content to handle newlines properly
    # Replace literal "\n" with actual newlines if they're not already processed
    if "\\n" in content and "\n" not in content:
        content = content.replace("\\n", "\n")
    
    console.print(Panel(f"[bold yellow]Action Proposed:[/bold yellow] Create file '{escape(path)}'",
                  # Escape path
                        title="[bold yellow]Confirmation Required[/bold yellow]", expand=False))
    # Strip leading/trailing whitespace, including newlines
    content_stripped = content.strip()
    console.print(Panel(Syntax(content_stripped, "text", theme="monokai", line_numbers=True),
                  title="[bold yellow]File Content[/bold yellow]", expand=False))

    # Use custom confirmation prompt with thinking_status
    # Escape path
    if ask_confirmation(f"Do you want to proceed with creating '{escape(path)}'? (y/n)"):
        try:
            # Ensure parent directory exists if path includes directories
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content_stripped)  # Write stripped content
            console.print(Panel(
                # Escape path
                f"[bold green]Success:[/bold green] File '{escape(path)}' created.", expand=False))
            
            # Track this file create operation
            if path not in agent_context["last_modified_files"]:
                agent_context["last_modified_files"].insert(0, path)
                # Keep the list to a reasonable size
                if len(agent_context["last_modified_files"]) > 5:
                    agent_context["last_modified_files"].pop()
            
            # Add this action to context
            agent_context["last_actions"].insert(0, {"action": "create_file", "path": path})
            if len(agent_context["last_actions"]) > 10:
                agent_context["last_actions"].pop()
            
            # Analyze code for issues if it's a code file
            analyze_code(path, content_stripped)
                
            return f"File '{path}' created successfully."
        except Exception as e:
            console.print(Panel(
                # Escape path
                f"[bold red]Error:[/bold red] Failed to create file '{escape(path)}'. Reason: {e}", expand=False))
            return f"Failed to create file '{path}'. Reason: {e}"
    else:
        console.print(Panel(
            "[bold red]Aborted:[/bold red] File creation cancelled by user.", expand=False))
        return f"File creation for '{path}' cancelled by user."


@tool
def write_file(path: str, content: str) -> str:
    """Overwrites an existing file at the specified path with the given content. Requires user confirmation."""
    # Get the global thinking_status if it exists
    global thinking_status, agent_context

    # Process content to handle newlines properly
    # Replace literal "\n" with actual newlines if they're not already processed
    if "\\n" in content and "\n" not in content:
        content = content.replace("\\n", "\n")

    console.print(Panel(f"[bold yellow]Action Proposed:[/bold yellow] Overwrite file '{escape(path)}'",
                  # Escape path
                        title="[bold yellow]Confirmation Required[/bold yellow]", expand=False))
    try:
        with open(path, 'r', encoding='utf-8') as f:
            old_content = f.read()
        console.print(Panel(Syntax(old_content, "text", theme="monokai", line_numbers=True),
                      title="[bold yellow]Current File Content[/bold yellow]", expand=False))
    except FileNotFoundError:
        console.print(Panel(
            # Escape path
            f"[bold red]Warning:[/bold red] File '{escape(path)}' not found. This operation will create it instead of overwriting.", expand=False))
        old_content = None  # Indicate that there was no old content
    except Exception as e:
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] Could not read current file content for '{escape(path)}'. Reason: {e}", expand=False))
        old_content = None  # Indicate failure to read old content

    # Strip leading/trailing whitespace, including newlines
    content_stripped = content.strip()
    console.print(Panel(Syntax(content_stripped, "text", theme="monokai", line_numbers=True),
                  title="[bold yellow]New File Content[/bold yellow]", expand=False))

    # Use custom confirmation prompt with thinking_status
    # Escape path
    if ask_confirmation(f"Do you want to proceed with overwriting '{escape(path)}'? (y/n)"):
        try:
            # Ensure parent directory exists if path includes directories
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content_stripped)  # Write stripped content
            console.print(Panel(
                # Escape path
                f"[bold green]Success:[/bold green] File '{escape(path)}' overwritten.", expand=False))
            
            # Track this file write operation
            if path not in agent_context["last_modified_files"]:
                agent_context["last_modified_files"].insert(0, path)
                # Keep the list to a reasonable size
                if len(agent_context["last_modified_files"]) > 5:
                    agent_context["last_modified_files"].pop()
            
            # Add this action to context
            agent_context["last_actions"].insert(0, {"action": "write_file", "path": path})
            if len(agent_context["last_actions"]) > 10:
                agent_context["last_actions"].pop()
            
            # Analyze code for issues if it's a code file
            analyze_code(path, content_stripped)
                
            return f"File '{path}' overwritten successfully."
        except Exception as e:
            console.print(Panel(
                # Escape path
                f"[bold red]Error:[/bold red] Failed to overwrite file '{escape(path)}'. Reason: {e}", expand=False))
            return f"Failed to overwrite file '{path}'. Reason: {e}"
    else:
        console.print(Panel(
            "[bold red]Aborted:[/bold red] File overwrite cancelled by user.", expand=False))
        return f"File overwrite for '{path}' cancelled by user."


@tool
def rename_path(old_path: str, new_path: str) -> str:
    """Renames a file or directory from old_path to new_path. Requires user confirmation."""
    # Get the global thinking_status if it exists
    global thinking_status

    console.print(Panel(f"[bold yellow]Action Proposed:[/bold yellow] Rename '{escape(old_path)}' to '{escape(new_path)}'",
                  # Escape paths
                        title="[bold yellow]Confirmation Required[/bold yellow]", expand=False))

    # Check if old_path exists
    if not os.path.exists(old_path):
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] Source path '{escape(old_path)}' not found.", expand=False))
        return f"Error: Source path '{old_path}' not found."

    # Check if new_path already exists (optional, but good practice)
    if os.path.exists(new_path):
        console.print(Panel(
            # Escape path
            f"[bold red]Warning:[/bold red] Destination path '{escape(new_path)}' already exists. It will be overwritten.", expand=False))
        # You might want to add a specific confirmation for overwriting,
        # but for simplicity, the main confirmation covers it.

    # Use custom confirmation prompt with thinking_status
    # Escape paths
    if ask_confirmation(f"Do you want to proceed with renaming '{escape(old_path)}' to '{escape(new_path)}'? (y/n)"):
        try:
            # Ensure parent directory of new_path exists
            os.makedirs(os.path.dirname(new_path) or '.', exist_ok=True)
            os.rename(old_path, new_path)
            console.print(Panel(
                # Escape paths
                f"[bold green]Success:[/bold green] Renamed '{escape(old_path)}' to '{escape(new_path)}'.", expand=False))
            return f"Successfully renamed '{old_path}' to '{new_path}'."
        except FileNotFoundError:  # This should be caught by the initial check, but kept for robustness
            console.print(Panel(
                # Escape path
                f"[bold red]Error:[/bold red] Source path '{escape(old_path)}' not found during rename.", expand=False))
            return f"Error: Source path '{old_path}' not found during rename."
        # If new_path exists and is a non-empty directory (os.rename cannot overwrite non-empty dirs)
        except FileExistsError:
            console.print(Panel(
                # Escape path
                f"[bold red]Error:[/bold red] Destination path '{escape(new_path)}' exists and cannot be overwritten (e.g., non-empty directory).", expand=False))
            return f"Error: Destination path '{new_path}' exists and cannot be overwritten (e.g., non-empty directory)."
        except PermissionError:
            console.print(Panel(
                # Escape paths
                f"[bold red]Error:[/bold red] Permission denied to rename '{escape(old_path)}' to '{escape(new_path)}'.", expand=False))
            return f"Error: Permission denied to rename '{old_path}' to '{new_path}'."
        except Exception as e:
            console.print(Panel(
                f"[bold red]Error:[/bold red] An unexpected error occurred while renaming. Reason: {e}", expand=False))
            return f"Error: An unexpected error occurred while renaming. Reason: {e}"
    else:
        console.print(Panel(
            "[bold red]Aborted:[/bold red] Rename operation cancelled by user.", expand=False))
        return f"Rename operation from '{old_path}' to '{new_path}' cancelled by user."


@tool
def delete_path(path: str) -> str:
    """Deletes a file or directory at the specified path. Requires user confirmation. USE WITH CAUTION!"""
    # Get the global thinking_status if it exists
    global thinking_status

    # Check if path exists
    if not os.path.exists(path):
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] Path '{escape(path)}' not found.", expand=False))
        return f"Error: Path '{path}' not found."

    # Provide more details about what's being deleted
    if os.path.isdir(path):
        # Count files/folders in directory
        try:
            item_count = sum(len(files) + len(dirs)
                             for _, dirs, files in os.walk(path))
            dir_type = "directory"
            console.print(Panel(
                f"[bold red]⚠️ CAUTION:[/bold red] You are about to delete a directory containing approximately {item_count} items!", title=f"[bold red]Delete {dir_type}[/bold red]", expand=False))
        except Exception:
            dir_type = "directory"
            console.print(Panel(f"[bold red]⚠️ CAUTION:[/bold red] You are about to delete a directory!",
                          title=f"[bold red]Delete {dir_type}[/bold red]", expand=False))
    else:
        # It's a file, so get its size
        try:
            file_size = os.path.getsize(path)
            size_str = f"{file_size / 1024:.1f} KB" if file_size >= 1024 else f"{file_size} bytes"
            console.print(Panel(
                f"[bold yellow]You are about to delete a file of size {size_str}[/bold yellow]", title="[bold red]Delete File[/bold red]", expand=False))
        except Exception:
            console.print(Panel(f"[bold yellow]You are about to delete a file[/bold yellow]",
                          title="[bold red]Delete File[/bold red]", expand=False))

    # Use custom confirmation prompt with thinking_status
    # Escape path
    if ask_confirmation(f"Are you SURE you want to delete '{escape(path)}'? This CANNOT be undone! (y/n)"):
        try:
            if os.path.isdir(path):
                # Use shutil.rmtree for directories
                shutil.rmtree(path)
                console.print(Panel(
                    # Escape path
                    f"[bold green]Success:[/bold green] Directory '{escape(path)}' and all its contents deleted.", expand=False))
                return f"Directory '{path}' and all its contents deleted successfully."
            else:
                # Use os.remove for files
                os.remove(path)
                console.print(Panel(
                    # Escape path
                    f"[bold green]Success:[/bold green] File '{escape(path)}' deleted.", expand=False))
                return f"File '{path}' deleted successfully."
        except PermissionError:
            console.print(Panel(
                # Escape path
                f"[bold red]Error:[/bold red] Permission denied to delete '{escape(path)}'.", expand=False))
            return f"Error: Permission denied to delete '{path}'."
        except Exception as e:
            console.print(Panel(
                # Escape path
                f"[bold red]Error:[/bold red] An unexpected error occurred while deleting '{escape(path)}'. Reason: {e}", expand=False))
            return f"Error: An unexpected error occurred while deleting '{path}'. Reason: {e}"
    else:
        console.print(Panel(
            "[bold red]Aborted:[/bold red] Delete operation cancelled by user.", expand=False))
        return f"Delete operation for '{path}' cancelled by user."


@tool
def run_command(command: str) -> str:
    """Runs a terminal command and returns its output."""
    global thinking_status

    console.print(Panel(f"[bold yellow]Running Command:[/bold yellow] {escape(command)}",
                  # Escape command
                        title="[bold yellow]Terminal Command[/bold yellow]", expand=False))

    # Stop thinking indicator before asking for confirmation
    if thinking_status:
        thinking_status.stop()

    # Use custom confirmation prompt
    # Escape command
    if not ask_confirmation(f"Do you want to run the command: '{escape(command)}'? (y/n)"):
        console.print(Panel(
            "[bold red]Aborted:[/bold red] Command execution cancelled by user.", expand=False))
        return "Command execution cancelled by user."

    try:
        # Run the command and capture output
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Use console status to show that command is running
        with Status("[bold yellow]Running command...[/bold yellow]", spinner="dots", console=console):
            stdout, stderr = process.communicate()

        # Print command output with proper syntax highlighting
        if stdout:
            # Try to determine the language from the command
            language = "bash"  # Default to bash
            if command.startswith("python"):
                language = "python"
            elif any(cmd in command for cmd in ["node", "npm", "js"]):
                language = "javascript"

            console.print(Panel(Syntax(stdout, language, theme="monokai"),
                          title="[bold green]Command Output[/bold green]", expand=False))

        # Print stderr if there was any
        if stderr:
            console.print(Panel(Syntax(stderr, "bash", theme="monokai"),
                          title="[bold red]Command Error Output[/bold red]", expand=False))

        # Print return code
        exit_code = process.returncode
        if exit_code == 0:
            console.print(Panel(
                f"[bold green]Command completed successfully with exit code: {exit_code}[/bold green]", expand=False))
        else:
            console.print(Panel(
                f"[bold red]Command failed with exit code: {exit_code}[/bold red]", expand=False))

        # Return combined output
        result = f"Exit Code: {exit_code}\n"
        if stdout:
            result += f"Standard Output:\n{stdout}\n"
        if stderr:
            result += f"Standard Error:\n{stderr}\n"
        return result.strip()

    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/bold red] Failed to execute command. Reason: {e}", expand=False))
        return f"Failed to execute command. Reason: {e}"


@tool
def modify_file(path: str, search_text: str, replace_text: str) -> str:
    """Modifies a file by replacing occurrences of search_text with replace_text. Requires user confirmation."""
    # Get the global thinking_status if it exists
    global thinking_status, agent_context

    # Check if file exists
    if not os.path.exists(path):
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] File '{escape(path)}' not found.", expand=False))
        return f"Error: File '{path}' not found."

    # Check if it's a file, not a directory
    if not os.path.isfile(path):
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] Path '{escape(path)}' is not a file.", expand=False))
        return f"Error: Path '{path}' is not a file."

    # First read the file content
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] Could not decode file '{escape(path)}' with utf-8. It might be a binary file.", expand=False))
        return f"Error: Could not decode file '{path}' with utf-8. It might be a binary file."
    except Exception as e:
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] An unexpected error occurred while reading '{escape(path)}'. Reason: {e}", expand=False))
        return f"Error: An unexpected error occurred while reading '{path}'. Reason: {e}"

    # Count occurrences
    occurrences = content.count(search_text)

    if occurrences == 0:
        console.print(Panel(
            # Escape path
            f"[bold yellow]Warning:[/bold yellow] The exact search text was not found in '{escape(path)}'.", expand=False))

        # Try to find similar text that might be what the user is looking for
        similar_texts = find_similar_text(content, search_text)
        if similar_texts:
            console.print(Panel(
                "[bold yellow]Similar text found in the file:[/bold yellow]", expand=False))
            for i, text in enumerate(similar_texts, 1):
                text_to_display = text.strip()
                console.print(
                    f"[bold yellow]{i}.[/bold yellow] {escape(text_to_display)}")

        return f"No occurrences of the exact search text found in '{path}'. Please check the file content and try again with a more precise search text."

    # Calculate the new content
    new_content = content.replace(search_text, replace_text)

    # Show a preview with explanation
    console.print(Panel(f"[bold yellow]Action Proposed:[/bold yellow] Modify file '{escape(path)}'",
                  # Escape path
                        title="[bold yellow]Confirmation Required[/bold yellow]", expand=False))
    console.print(
        f"[bold yellow]Will replace {occurrences} occurrence(s) of the following text:[/bold yellow]")
    console.print(Panel(Syntax(search_text, "text", theme="monokai"),
                  title="[bold yellow]Search Text[/bold yellow]", expand=False))
    console.print("[bold yellow]With this text:[/bold yellow]")
    console.print(Panel(Syntax(replace_text, "text", theme="monokai"),
                  title="[bold yellow]Replace Text[/bold yellow]", expand=False))

    # Use custom confirmation prompt with thinking_status
    # Escape path
    if ask_confirmation(f"Do you want to proceed with modifying '{escape(path)}'? (y/n)"):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            console.print(Panel(
                # Escape path
                f"[bold green]Success:[/bold green] Modified {occurrences} occurrence(s) in file '{escape(path)}'.", expand=False))
            
            # Track this file modification operation
            if path not in agent_context["last_modified_files"]:
                agent_context["last_modified_files"].insert(0, path)
                # Keep the list to a reasonable size
                if len(agent_context["last_modified_files"]) > 5:
                    agent_context["last_modified_files"].pop()
            
            # Add this action to context
            agent_context["last_actions"].insert(0, {"action": "modify_file", "path": path, "occurrences": occurrences})
            if len(agent_context["last_actions"]) > 10:
                agent_context["last_actions"].pop()
            
            # Analyze code for issues if it's a code file and auto verification is enabled
            if auto_verify_enabled:
                analyze_code(path, new_content)
                
            return f"Successfully modified {occurrences} occurrence(s) in file '{path}'."
        except PermissionError:
            console.print(Panel(
                # Escape path
                f"[bold red]Error:[/bold red] Permission denied to modify file '{escape(path)}'.", expand=False))
            return f"Error: Permission denied to modify file '{path}'."
        except Exception as e:
            console.print(Panel(
                # Escape path
                f"[bold red]Error:[/bold red] An unexpected error occurred while modifying '{escape(path)}'. Reason: {e}", expand=False))
            return f"Error: An unexpected error occurred while modifying '{path}'. Reason: {e}"
    else:
        console.print(Panel(
            "[bold red]Aborted:[/bold red] File modification cancelled by user.", expand=False))
        return f"File modification for '{path}' cancelled by user."


def find_similar_text(content, search_text, max_results=3):
    """Find text in content that is similar to the search text."""
    # Split content into lines
    lines = content.splitlines()

    # Simple similarity score - how many characters match in sequence
    def similarity_score(text1, text2):
        score = 0
        min_len = min(len(text1), len(text2))
        for i in range(min_len):
            if text1[i] == text2[i]:
                score += 1
            else:
                break
        return score

    # Try to find similar text
    candidates = []
    search_stripped = search_text.strip()

    # Look for partial matches in each line
    for line in lines:
        line_stripped = line.strip()
        if search_stripped in line_stripped:
            candidates.append(line)
            continue

        # Try matching the beginning of search text with parts of this line
        words = line_stripped.split()
        for i in range(len(words)):
            partial = " ".join(words[i:])
            if partial and search_stripped.startswith(partial):
                candidates.append(line)
                break

            # Try matching the end of search text with parts of this line
            partial = " ".join(words[:i+1])
            if partial and search_stripped.endswith(partial):
                candidates.append(line)
                break

    # Sort candidates by similarity and uniqueness
    unique_candidates = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    # Return top matches, up to max_results
    return unique_candidates[:max_results]

def parse_diff_string(diff_string):
    """Parses a unified diff string and returns a patch set."""
    try:
        # First attempt basic validation of the diff format
        lines = diff_string.strip().split('\n')
        if not any(line.startswith('---') for line in lines) or not any(line.startswith('+++') for line in lines):
            console.print("[bold yellow]Warning: Diff format is missing file headers (--- or +++).[/bold yellow]")
        
        if not any(line.startswith('@@') for line in lines):
            console.print("[bold yellow]Warning: Diff format is missing hunk headers (@@).[/bold yellow]")
        
        # Try to parse using unidiff
        patch_set = unidiff.PatchSet.from_string(diff_string)
        
        # Validate the patch set - check that it's not empty
        if len(patch_set) == 0:
            console.print("[bold yellow]Warning: Diff parsed successfully but contains no patches.[/bold yellow]")
            return None
            
        return patch_set
    except Exception as e:
        console.print(f"[bold red]Error parsing diff: {str(e)}[/bold red]")
        
        # Fallback for manual parsing
        try:
            # Instead of using a temporary file, we'll manually create a PatchSet
            # This is a very simple implementation that just handles basic add/remove lines
            manual_patch = []
            in_hunk = False
            hunk_lines = []
            
            for line in diff_string.strip().split('\n'):
                if line.startswith('@@'):
                    in_hunk = True
                    hunk_lines = [line]
                elif in_hunk and (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                    hunk_lines.append(line)
            
            if hunk_lines:
                # Build a simple diff that we can apply manually
                return {
                    'manual': True,
                    'lines': diff_string.strip().split('\n')
                }
            
            return None
        except Exception as fallback_e:
            console.print(f"[bold red]Error during fallback diff parsing: {fallback_e}[/bold red]")
            return None


def apply_diff_to_content(content, patch_set):
    """Applies a patch set to the content and returns the new content."""
    # Handle the manual patch format
    if isinstance(patch_set, dict) and patch_set.get('manual'):
        return apply_manual_diff(content, patch_set['lines'])
    
    # Preserve original content in case of errors
    original_content = content
    
    try:
        # Split content into lines - ensure proper line handling
        lines = content.splitlines(True)  # Keep the line endings
        
        applied_hunks = 0
        
        # Apply each patch in the patch set
        for patched_file in patch_set:
            for hunk in patched_file:
                # Calculate the line range for this hunk
                start_line = hunk.source_start - 1  # 0-indexed
                end_line = start_line + hunk.source_length
                
                # Check bounds
                if start_line < 0 or end_line > len(lines):
                    console.print(f"[bold yellow]Warning: Hunk at line {start_line+1} is out of bounds for the file (max {len(lines)}).[/bold yellow]")
                    continue
                
                # Get the lines that should be replaced
                source_lines = lines[start_line:end_line]
                
                # Preserve line endings in the comparison
                source_text = ''.join(source_lines)
                hunk_source = hunk.source
                
                # Normalize hunk source to match file's line endings if needed
                if '\r\n' in content and '\n' in hunk_source and '\r\n' not in hunk_source:
                    hunk_source = hunk_source.replace('\n', '\r\n')
                elif '\n' in content and '\r\n' in hunk_source:
                    hunk_source = hunk_source.replace('\r\n', '\n')
                
                # Debug output for troubleshooting
                if len(source_lines) > 0:
                    console.print(f"[dim]Comparing hunk at line {start_line+1}:[/dim]")
                
                # Simplify the comparison for logging purposes
                source_for_log = source_text.replace('\n', '\\n').replace('\r', '\\r')
                hunk_for_log = hunk_source.replace('\n', '\\n').replace('\r', '\\r')
                if len(source_for_log) > 50:
                    source_for_log = source_for_log[:47] + "..."
                if len(hunk_for_log) > 50:
                    hunk_for_log = hunk_for_log[:47] + "..."
                
                console.print(f"[dim]Source: '{source_for_log}'[/dim]")
                console.print(f"[dim]Hunk: '{hunk_for_log}'[/dim]")
                
                matched = False
                
                # First try exact match
                if source_text == hunk_source:
                    matched = True
                    match_type = "exact"
                # Try with normalized whitespace
                elif source_text.strip() == hunk_source.strip():
                    matched = True
                    match_type = "whitespace-trimmed"
                # Try with very flexible matching (no whitespace)
                elif ''.join(source_text.split()) == ''.join(hunk_source.split()):
                    matched = True
                    match_type = "no-whitespace"
                
                if matched:
                    # Replace the lines
                    target_lines = hunk.target.splitlines(True)
                    
                    console.print(f"[bold green]Matched hunk at line {start_line+1} with {match_type} matching.[/bold green]")
                    
                    if not target_lines and source_lines:
                        # This is a deletion
                        del lines[start_line:end_line]
                    else:
                        # Replacement or addition
                        lines[start_line:end_line] = target_lines
                    
                    applied_hunks += 1
                else:
                    # Try to manually split and apply for better matching
                    console.print(f"[bold yellow]Warning: Hunk at line {start_line+1} does not match source content.[/bold yellow]")
                    console.print("[yellow]Attempting flexible matching...[/yellow]")
                    
                    # Check if this is a simple addition
                    if hunk.source.strip() == '' and hunk.target.strip() != '':
                        # This is an addition
                        target_lines = hunk.target.splitlines(True)
                        lines[start_line:start_line] = target_lines
                        console.print(f"[bold green]Applied addition at line {start_line+1}.[/bold green]")
                        applied_hunks += 1
                    else:
                        console.print("[bold red]Could not apply hunk with flexible matching.[/bold red]")
        
        # Join the lines back into a single string
        new_content = ''.join(lines)
        
        # If we didn't apply any hunks, return the original content
        if applied_hunks == 0:
            return original_content, applied_hunks
        
        return new_content, applied_hunks
    except Exception as e:
        console.print(f"[bold red]Error applying patches: {e}[/bold red]")
        console.print("[yellow]Returning original file content due to error.[/yellow]")
        return original_content, 0


def apply_manual_diff(content, diff_lines):
    """Apply a diff manually line by line for cases where the unidiff library fails."""
    # Preserve original content
    original_content = content
    
    try:
        content_lines = content.splitlines(True)  # Keep the line endings
        result_lines = content_lines.copy()
        applied_hunks = 0
        
        # Parse the diff to find hunks
        hunks = []
        current_hunk = None
        
        for line in diff_lines:
            if line.startswith('@@'):
                # Start a new hunk
                if current_hunk:
                    hunks.append(current_hunk)
                
                # Parse the hunk header: @@ -start,count +start,count @@
                try:
                    header_parts = line.split('@@')[1].strip().split(' ')
                    source_part = header_parts[0].strip()
                    source_info = source_part[1:].split(',')  # Remove the '-' and split by comma
                    source_start = int(source_info[0])
                    
                    # Create new hunk
                    current_hunk = {
                        'source_start': source_start,
                        'lines': [line],
                        'content': []
                    }
                except Exception as e:
                    console.print(f"[bold yellow]Warning: Failed to parse hunk header: {line}. {e}[/bold yellow]")
                    current_hunk = None
            elif current_hunk is not None:
                # Add line to current hunk
                current_hunk['lines'].append(line)
                current_hunk['content'].append(line)
        
        # Add the last hunk if there is one
        if current_hunk:
            hunks.append(current_hunk)
        
        if not hunks:
            console.print("[bold yellow]Warning: No valid hunks found in diff.[/bold yellow]")
            return original_content, 0
        
        # Print some debug info
        console.print(f"[dim]Found {len(hunks)} hunks in manual diff.[/dim]")
        
        # Process each hunk
        for hunk_idx, hunk in enumerate(hunks):
            source_start = hunk['source_start'] - 1  # Convert to 0-indexed
            
            console.print(f"[dim]Processing hunk {hunk_idx+1} at line {source_start+1}...[/dim]")
            
            # Find context lines to locate the insertion point
            context_before = []
            context_after = []
            remove_lines = []
            add_lines = []
            
            for line in hunk['content']:
                if line.startswith(' '):
                    if not remove_lines and not add_lines:
                        context_before.append(line[1:])  # Remove the space prefix
                    else:
                        context_after.append(line[1:])  # Remove the space prefix
                elif line.startswith('-'):
                    remove_lines.append(line[1:])  # Remove the - prefix
                elif line.startswith('+'):
                    add_lines.append(line[1:])  # Remove the + prefix
            
            # Try to find the insertion point based on context
            insertion_point = find_insertion_point(content_lines, context_before, source_start)
            
            if insertion_point >= 0:
                console.print(f"[dim]Found insertion point at line {insertion_point+1}.[/dim]")
                
                # Validate that remove_lines match the actual content
                if remove_lines:
                    matched = True
                    for i, remove_line in enumerate(remove_lines):
                        if insertion_point + len(context_before) + i >= len(content_lines):
                            matched = False
                            break
                            
                        actual_line = content_lines[insertion_point + len(context_before) + i]
                        if actual_line.rstrip() != remove_line.rstrip():
                            matched = False
                            
                            # Debug output
                            console.print(f"[dim]Line to remove: '{remove_line.rstrip()}'[/dim]")
                            console.print(f"[dim]Actual line: '{actual_line.rstrip()}'[/dim]")
                            break
                    
                    if not matched:
                        console.print("[bold yellow]Warning: Lines to remove don't match actual content.[/bold yellow]")
                        console.print("[yellow]Attempting flexible matching...[/yellow]")
                        
                        # Try more flexible matching
                        # ... Flexible matching logic would go here ...
                        
                        console.print("[bold red]Could not apply hunk with flexible matching.[/bold red]")
                        continue
                        
                # Apply the changes
                new_lines = []
                
                # Add lines before the change
                new_lines.extend(result_lines[:insertion_point + len(context_before)])
                
                # Add the new lines (if any)
                new_lines.extend(add_lines)
                
                # Add lines after the change
                new_lines.extend(result_lines[insertion_point + len(context_before) + len(remove_lines):])
                
                # Update result_lines for the next hunk
                result_lines = new_lines
                applied_hunks += 1
                console.print(f"[bold green]Successfully applied hunk {hunk_idx+1}.[/bold green]")
            else:
                console.print(f"[bold yellow]Warning: Could not find insertion point for hunk {hunk_idx+1}.[/bold yellow]")
        
        if applied_hunks == 0:
            console.print("[bold yellow]Warning: No hunks could be applied.[/bold yellow]")
            return original_content, 0
            
        return ''.join(result_lines), applied_hunks
    except Exception as e:
        console.print(f"[bold red]Error in manual diff application: {e}[/bold red]")
        console.print("[yellow]Returning original file content due to error.[/yellow]")
        return original_content, 0


def find_insertion_point(content_lines, context_before, hint_line):
    """Find the insertion point in the content based on context lines."""
    if not context_before:
        return hint_line  # Use the hint if no context
    
    # Debug output
    context_str = ''.join(context_before)
    if len(context_str) > 50:
        context_display = context_str[:47] + "..."
    else:
        context_display = context_str
        
    console.print(f"[dim]Looking for context: '{context_display}'[/dim]")
    console.print(f"[dim]Hint line: {hint_line+1}[/dim]")
    
    # Search around the hint line first with exact matching
    search_range = 10  # Look 10 lines before and after the hint
    start = max(0, hint_line - search_range)
    end = min(len(content_lines), hint_line + search_range)
    
    for i in range(start, end):
        if i + len(context_before) <= len(content_lines):
            candidate = ''.join(content_lines[i:i+len(context_before)])
            if candidate == context_str:
                console.print(f"[dim]Found exact match at line {i+1}[/dim]")
                return i
            
    # Try with trimmed whitespace
    for i in range(start, end):
        if i + len(context_before) <= len(content_lines):
            candidate = ''.join(content_lines[i:i+len(context_before)])
            if candidate.strip() == context_str.strip():
                console.print(f"[dim]Found trimmed whitespace match at line {i+1}[/dim]")
                return i
    
    # If not found near the hint, try without whitespace
    for i in range(len(content_lines)):
        if i + len(context_before) <= len(content_lines):
            candidate = ''.join(content_lines[i:i+len(context_before)])
            # Compare without whitespace
            if ''.join(candidate.split()) == ''.join(context_str.split()):
                console.print(f"[dim]Found no-whitespace match at line {i+1}[/dim]")
                return i
    
    # If still not found, try line by line matching to find partial matches
    best_match = -1
    best_score = -1
    
    for i in range(len(content_lines) - len(context_before) + 1):
        score = 0
        for j, context_line in enumerate(context_before):
            content_line = content_lines[i + j]
            if content_line.strip() == context_line.strip():
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = i
    
    if best_score > len(context_before) / 2:
        console.print(f"[dim]Found partial match ({best_score}/{len(context_before)} lines) at line {best_match+1}[/dim]")
        return best_match
    
    # If still not found, just use the hint
    console.print(f"[dim]No match found, using hint line {hint_line+1}[/dim]")
    return hint_line


@tool
def apply_diff(path: str, diff_content: str) -> str:
    """Applies a diff to a file. The diff should be in unified diff format. Requires user confirmation."""
    # Get the global thinking_status if it exists
    global thinking_status, agent_context

    # Check if file exists
    if not os.path.exists(path):
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] File '{escape(path)}' not found.", expand=False))
        return f"Error: File '{path}' not found."

    # Check if it's a file, not a directory
    if not os.path.isfile(path):
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] Path '{escape(path)}' is not a file.", expand=False))
        return f"Error: Path '{path}' is not a file."

    # First read the file content
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] Could not decode file '{escape(path)}' with utf-8. It might be a binary file.", expand=False))
        return f"Error: Could not decode file '{path}' with utf-8. It might be a binary file."
    except Exception as e:
        console.print(Panel(
            # Escape path
            f"[bold red]Error:[/bold red] An unexpected error occurred while reading '{escape(path)}'. Reason: {e}", expand=False))
        return f"Error: An unexpected error occurred while reading '{path}'. Reason: {e}"

    # Preserve original content before any modification
    original_content = content

    # Log the diff for debugging
    console.print("[bold blue]Processing diff...[/bold blue]")
    
    # Try the alternative approach first - Generate a clean diff file with GNU diff if needed
    try:
        # Write original content to a temp file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.original') as src_file:
            src_file.write(content)
            src_path = src_file.name
        
        # Apply the diff manually using string operations
        lines = content.splitlines()
        new_lines = lines.copy()
        
        # Process the diff line by line
        diff_lines = diff_content.strip().split('\n')
        
        # Look for hunk headers
        hunk_pattern = re.compile(r'^@@ -(\d+),(\d+) \+(\d+),(\d+) @@')
        in_hunk = False
        hunk_start = 0
        hunk_length = 0
        hunk_offset = 0
        added_lines = []
        removed_lines = []
        
        for line in diff_lines:
            # Skip file headers
            if line.startswith('---') or line.startswith('+++'):
                continue
                
            # Parse hunk headers
            hunk_match = hunk_pattern.match(line)
            if hunk_match:
                # Apply the previous hunk if we were in one
                if in_hunk:
                    # Apply removals first
                    for i, line_num in enumerate(sorted(removed_lines, reverse=True)):
                        del new_lines[line_num - 1 + hunk_offset]
                        hunk_offset -= 1
                    
                    # Then apply additions
                    for line_num, content in sorted(added_lines):
                        new_lines.insert(line_num - 1 + hunk_offset, content)
                        hunk_offset += 1
                
                # Reset for the new hunk
                in_hunk = True
                hunk_start = int(hunk_match.group(1))
                hunk_length = int(hunk_match.group(2))
                added_lines = []
                removed_lines = []
                continue
            
            # Process content lines within a hunk
            if in_hunk:
                if line.startswith('-'):
                    # Line to remove
                    line_num = hunk_start + len(removed_lines)
                    removed_lines.append(line_num)
                elif line.startswith('+'):
                    # Line to add
                    line_num = hunk_start + len(added_lines)
                    added_lines.append((line_num, line[1:]))
        
        # Apply any remaining hunk
        if in_hunk:
            # Apply removals first
            for i, line_num in enumerate(sorted(removed_lines, reverse=True)):
                if line_num - 1 + hunk_offset < len(new_lines):
                    del new_lines[line_num - 1 + hunk_offset]
                    hunk_offset -= 1
            
            # Then apply additions
            for line_num, content in sorted(added_lines):
                new_lines.insert(line_num - 1 + hunk_offset, content)
        
        # Join the lines with the original line endings
        if '\r\n' in content:
            new_content = '\r\n'.join(new_lines)
        else:
            new_content = '\n'.join(new_lines)
        
        # Show a preview with explanation
        console.print(Panel(f"[bold yellow]Action Proposed:[/bold yellow] Apply diff to file '{escape(path)}'",
                      # Escape path
                            title="[bold yellow]Confirmation Required[/bold yellow]", expand=False))
        
        # Show diff in a nice syntax-highlighted format
        console.print(Panel(Syntax(diff_content, "diff", theme="monokai"),
                     title="[bold yellow]Diff to Apply[/bold yellow]", expand=False))

        # Use custom confirmation prompt with thinking_status
        # Escape path
        if ask_confirmation(f"Do you want to proceed with applying this diff to '{escape(path)}'? (y/n)"):
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                console.print(Panel(
                    # Escape path
                    f"[bold green]Success:[/bold green] Applied diff to file '{escape(path)}'.", expand=False))
                
                # Track this file modification operation
                if path not in agent_context["last_modified_files"]:
                    agent_context["last_modified_files"].insert(0, path)
                    # Keep the list to a reasonable size
                    if len(agent_context["last_modified_files"]) > 5:
                        agent_context["last_modified_files"].pop()
                
                # Add this action to context
                agent_context["last_actions"].insert(0, {"action": "apply_diff", "path": path})
                if len(agent_context["last_actions"]) > 10:
                    agent_context["last_actions"].pop()
                
                # Analyze code for issues if it's a code file
                analyze_code(path, new_content)
                    
                return f"Successfully applied diff to file '{path}'."
            except PermissionError:
                console.print(Panel(
                    # Escape path
                    f"[bold red]Error:[/bold red] Permission denied to modify file '{escape(path)}'.", expand=False))
                return f"Error: Permission denied to modify file '{path}'."
            except Exception as e:
                console.print(Panel(
                    # Escape path
                    f"[bold red]Error:[/bold red] An unexpected error occurred while modifying '{escape(path)}'. Reason: {e}", expand=False))
                return f"Error: An unexpected error occurred while modifying '{path}'. Reason: {e}"
        else:
            console.print(Panel(
                "[bold red]Aborted:[/bold red] Diff application cancelled by user.", expand=False))
            return f"Diff application for '{path}' cancelled by user."
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/bold red] Failed to apply diff. Reason: {e}", expand=False))
        return f"Error: Failed to apply diff. Reason: {e}"


def generate_diff(original_content, new_content, file_path):
    """Generates a unified diff between original and new content."""
    original_lines = original_content.splitlines(True)
    new_lines = new_content.splitlines(True)
    
    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f'a/{file_path}',
        tofile=f'b/{file_path}',
        n=3  # Context lines
    )
    
    return ''.join(diff)

# --- Agent Setup ---


def setup_agent(api_key: str):
    """Sets up the LangChain agent with the Gemini model and tools."""
    global creative_mode
    
    # Set temperature based on creative mode
    temperature = 0.7 if creative_mode else 0
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17", google_api_key=api_key, temperature=temperature)

    # Tools list (order doesn't strictly matter for tool-calling models, but can aid readability)
    # Reorder to put modify_file before apply_diff to encourage its use
    tools = [list_directory, read_file, create_file, write_file,
             rename_path, delete_path, run_command, modify_file, apply_diff]

    # Define the agent prompt
    creative_addition = """
You are currently in CREATIVE MODE. This means:
1. You should suggest more innovative and novel features
2. For websites and UIs, create visually appealing designs with stylish animations and effects
3. Add creative flair to your implementations and suggestions
4. Take more creative liberties while still meeting the core requirements
5. Suggest additional features that would enhance the user experience
""" if creative_mode else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are Coda2, an AI assistant that can perform tasks related to code development, file system management, and terminal operations.
You can list directory contents, read files, create files, write to files, rename files/directories, delete files/directories, and run terminal commands. You can also modify specific text in files.{creative_addition}

You have access to the following tools:
- `list_directory(path: str = '.')`: Lists the contents of a directory. Useful for understanding the project structure.
- `read_file(path: str)`: Reads the content of a file. Useful for examining existing code or configuration.
- `create_file(path: str, content: str)`: Creates a new file with specified content.
- `write_file(path: str, content: str)`: Overwrites an existing file with specified content.
- `rename_path(old_path: str, new_path: str)`: Renames a file or directory.
- `delete_path(path: str)`: Deletes a file or directory. USE WITH EXTREME CAUTION!
- `run_command(command: str)`: Runs a terminal command and returns its output. **You can and should use this to execute terminal commands directly when needed.**
- `modify_file(path: str, search_text: str, replace_text: str)`: **Searches for specific text in a file and replaces it with new text.** This is the preferred way to make changes to files. The tool handles finding and replacing the exact text you specify.
- `apply_diff(path: str, diff_content: str)`: **Applies a diff to modify a file.** This is an alternative way to make changes to code files for complex modifications where search/replace isn't suitable.

## IMPORTANT FOR FILE MODIFICATIONS:
PREFER using modify_file (search/replace) for most code changes as it provides a simple and direct way to make modifications. The search/replace approach lets you precisely target the exact text that needs changing without having to worry about line numbers or diff syntax.

For most changes, search/replace is recommended because:
1. It's simpler and more straightforward
2. You can focus on the exact text to change rather than diff formatting
3. It works well across different file types
4. It's easier to understand what's being changed

You should:
- Read the file first with read_file
- Identify the exact text to search for
- Provide the replacement text
- Use modify_file to make the change

## Using the terminal:
You have full access to execute terminal commands through the `run_command` tool. This allows you to:
1. Run any bash/shell command the user would run manually
2. Execute programming language interpreters (python, node, etc.)
3. Use command-line tools (grep, find, git, etc.)
4. Install packages or dependencies
5. Start services or run applications

When using terminal commands:
- Always explain what a command will do before running it
- Use commands that provide verbose output when possible
- For potentially destructive operations, use safer versions (e.g., `rm -i` instead of just `rm`)
- You can chain multiple commands using && or ; as needed
- For commands that might run indefinitely (servers, watching files), add timeout limits or run in background

## Using search/replace for code changes:
When modifying code files, use the `modify_file` tool which provides search and replace functionality. Here's how:

1. Read the original file content using `read_file`.
2. Identify the exact text you want to replace.
3. Create the replacement text that should go in its place.
4. Use `modify_file` with the file path, search text, and replacement text.

This approach is precise because it targets the exact text you specify, and the tool will show you how many occurrences were found before making any changes.

## Alternative: Using diffs for complex changes
The `apply_diff` tool can be used for particularly complex changes where search/replace isn't suitable. A unified diff looks like this:
```
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,7 @@
 context line
 context line
-line to remove
+line to add
 context line
 context line
```

However, this approach is more complex and usually not necessary for most changes.

## When to use each file modification tool:
- Use `modify_file` for most code changes and text replacements (recommended approach)
- Only use `apply_diff` for complex multi-line changes where simple search/replace isn't feasible
- Use `write_file` when completely replacing a file's content
- Use `create_file` for new files

## How the search/replace system works:
The `modify_file` tool performs an exact string match and replacement in the target file. Here's what you need to know:

1. **Exact Matching**: The search operation performs EXACT string matching (including whitespace and newlines). It will not interpret regex patterns or special characters.

2. **Multiple Replacements**: If the search text appears multiple times in the file, ALL occurrences will be replaced simultaneously. The tool will show you the count of replacements before confirming.

3. **Preserving Formatting**: The tool preserves all formatting, indentation and whitespace outside of the matched text. Only the exact matched text is replaced.

4. **Multiline Support**: The search text can span multiple lines (including line breaks). This is especially useful for replacing code blocks.

5. **Helpful Suggestions**: If the exact search text isn't found, the tool will suggest similar text patterns that might be what you're looking for.

## Best practices for using search/replace:

1. **Always Read First**: Always use `read_file` to view the target file's content before attempting to modify it. This helps you identify the exact text to replace.

2. **Be Specific**: Include enough context in your search text to ensure you're only replacing what you intend to. For code, include unique identifiers, function names, or distinctive patterns.

3. **Include Whitespace Correctly**: Pay careful attention to spaces, tabs, and newlines in both search and replace text. Ensure indentation matches exactly.

4. **Partial Updates**: For large files, prefer targeting specific blocks rather than trying to replace the entire file content.

5. **Special Characters**: Be mindful of special characters (quotes, backslashes, etc.) as they will be matched literally.

6. **Line Endings**: The tool handles different line endings (\\n, \\r\\n) automatically, but be aware they may affect matching.

7. **Watch the Count**: Always check the replacement count before confirming to ensure you're not making more changes than intended.

When you need to perform an action:
1. Think step-by-step about the user's request and how the tools can help.
2. If the request involves modifying specific text in a file, use the `modify_file` tool and identify the exact text to search for and replace. If it involves creating or completely replacing a file, use `create_file` or `write_file`.
3. If the request involves interacting with the terminal, use the `run_command` tool.
4. If you need to inspect the file system, use `list_directory` or `read_file`.
5. Formulate a plan, explaining it clearly to the user.
6. Use the appropriate tool by calling the tool function with the required arguments. **Do not output file content or modification patterns directly in your response unless you are showing an example or explaining, but use the tools to make the changes.**
7. You must ALWAYS ask for user confirmation by using the tools themselves, as they are designed to prompt the user. The user expects to type 'y' or 'n' followed by Enter after a proposed action.
8. After a tool execution, analyze the tool's output (the string returned by the tool function) and continue the process if necessary, or provide a final response to the user based on the tool's result.
9. Respond conversationally and provide updates on your progress and findings.

IMPORTANT: For code modifications, use the `modify_file` tool (search/replace) as the primary approach for most changes.

Begin!"""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Create the tool-calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    return agent_executor

# --- Special Command Handlers ---


def display_help():
    """Display help information about special commands."""
    help_table = Table(title="Coda2 Special Commands")
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="green")

    help_table.add_row("/help", "Display this help information")
    help_table.add_row("/toggle-subthink",
                       "Toggle the subthinking feature on/off (plan before execution)")
    help_table.add_row("/ststeps <number>",
                       "Set the number of subthinking steps (max 50)")
    help_table.add_row("/creative-mode",
                       "Toggle creative mode for more innovative suggestions and designs")
    help_table.add_row("/analyze <file_path>",
                       "Analyze a specific file for code issues and possible fixes")
    help_table.add_row("/toggle-verify",
                       "Toggle automatic code verification after each agent execution")
    help_table.add_row("/toggle-trust",
                       "Toggle trust mode to allow file operations without confirmation")
    help_table.add_row("quit", "Exit the application")

    console.print(help_table)
    
    if subthinking_enabled:
        console.print(Panel("[bold green]Subthinking is currently enabled.[/bold green]\n\nIn subthinking mode, the agent will:\n1. Think through your request in multiple planning steps\n2. No tools will be used during planning steps\n3. Tools will only be used in the final execution", title="[bold green]Subthinking Information[/bold green]", expand=False))
    else:
        console.print(Panel("[bold yellow]Subthinking is currently disabled.[/bold yellow]\n\nEnable it with /toggle-subthink if you want the agent to plan carefully before executing actions.", title="[bold yellow]Subthinking Information[/bold yellow]", expand=False))
        
    if creative_mode:
        console.print(Panel("[bold green]Creative mode is currently enabled.[/bold green]\n\nIn creative mode, the agent will:\n1. Suggest more innovative features\n2. Create more visually appealing designs\n3. Add stylish animations and effects for websites\n4. Take more creative liberties with implementations", title="[bold green]Creative Mode Information[/bold green]", expand=False))
    else:
        console.print(Panel("[bold yellow]Creative mode is currently disabled.[/bold yellow]\n\nEnable it with /creative-mode if you want more innovative and beautiful designs.", title="[bold yellow]Creative Mode Information[/bold yellow]", expand=False))
        
    if trust_mode_enabled:
        console.print(Panel("[bold green]Trust mode is currently enabled.[/bold green]\n\nThe agent can now perform file operations without asking for confirmation.\nThis includes:\n1. Creating files\n2. Modifying files\n3. Renaming files\n4. Applying diffs\n\nNOTE: Terminal commands will still require confirmation.", expand=False))
        console.print(Panel("[bold yellow]⚠️ Warning:[/bold yellow] The agent can now make changes to your files without asking. Use with caution.", expand=False))
    else:
        console.print(Panel("[bold yellow]Trust mode is currently disabled.[/bold yellow]\n\nEnable it with /toggle-trust if you want to allow the agent to make file changes without confirmation.", title="[bold yellow]Trust Mode Information[/bold yellow]", expand=False))


def toggle_subthinking():
    """Toggle the subthinking feature on or off."""
    global subthinking_enabled
    subthinking_enabled = not subthinking_enabled
    state = "enabled" if subthinking_enabled else "disabled"
    
    if subthinking_enabled:
        console.print(Panel(f"[bold green]Subthinking is now {state}.[/bold green]\n\nIn subthinking mode, the agent will:\n1. Think through the request in {subthinking_steps} steps without using tools\n2. Plan the required actions and code changes\n3. Only use tools in the final execution step", expand=False))
    else:
        console.print(Panel(f"[bold green]Subthinking is now {state}.[/bold green]", expand=False))


def toggle_creative_mode():
    """Toggle the creative mode feature on or off."""
    global creative_mode
    creative_mode = not creative_mode
    state = "enabled" if creative_mode else "disabled"
    
    if creative_mode:
        console.print(Panel(f"[bold green]Creative mode is now {state}.[/bold green]\n\nIn creative mode, the agent will:\n1. Suggest more innovative features\n2. Create more visually appealing designs\n3. Add stylish animations and effects for websites\n4. Take more creative liberties with implementations", expand=False))
    else:
        console.print(Panel(f"[bold green]Creative mode is now {state}.[/bold green]", expand=False))


def set_subthinking_steps(steps_str):
    """Set the number of subthinking steps."""
    global subthinking_steps
    try:
        steps = int(steps_str)
        if steps < 1:
            console.print(Panel(
                "[bold red]Error:[/bold red] Number of steps must be at least 1.", expand=False))
            return
        if steps > 50:
            console.print(Panel(
                "[bold red]Error:[/bold red] Maximum number of steps is 50.", expand=False))
            steps = 50

        subthinking_steps = steps
        console.print(Panel(
            f"[bold green]Subthinking steps set to {subthinking_steps}.[/bold green]\n\nThe agent will use {subthinking_steps} planning steps before execution:\n1. Analysis - Understand the request\n2. Planning - Create a detailed plan\n{'' if steps <= 2 else '3+. Refinement - Refine the approach'}", expand=False))
    except ValueError:
        console.print(Panel(
            "[bold red]Error:[/bold red] Please provide a valid number for subthinking steps.", expand=False))


def toggle_auto_verification():
    """Toggle the automatic code verification feature on or off."""
    global auto_verify_enabled
    auto_verify_enabled = not auto_verify_enabled
    state = "enabled" if auto_verify_enabled else "disabled"
    
    if auto_verify_enabled:
        console.print(Panel(f"[bold green]Automatic code verification is now {state}.[/bold green]\n\nAfter each agent execution, modified files will be analyzed for:\n1. Duplicated code blocks\n2. Syntax errors\n3. Logic errors or bugs\n4. Style issues and anti-patterns\n5. Security vulnerabilities", expand=False))
    else:
        console.print(Panel(f"[bold green]Automatic code verification is now {state}.[/bold green]\n\nYou can still analyze files manually using the /analyze command.", expand=False))


def toggle_trust_mode():
    """Toggle the trust mode feature on or off."""
    global trust_mode_enabled
    trust_mode_enabled = not trust_mode_enabled
    state = "enabled" if trust_mode_enabled else "disabled"
    
    if trust_mode_enabled:
        console.print(Panel(f"[bold green]Trust mode is now {state}.[/bold green]\n\nThe agent can now perform file operations without asking for confirmation.\nThis includes:\n1. Creating files\n2. Modifying files\n3. Renaming files\n4. Applying diffs\n\nNOTE: Terminal commands will still require confirmation.", expand=False))
        console.print(Panel("[bold yellow]⚠️ Warning:[/bold yellow] The agent can now make changes to your files without asking. Use with caution.", expand=False))
    else:
        console.print(Panel(f"[bold green]Trust mode is now {state}.[/bold green]\n\nThe agent will now ask for confirmation before performing any file operations.", expand=False))


def process_special_command(command):
    """Process special commands starting with /."""
    if command == "/help":
        display_help()
        return True
    elif command == "/toggle-subthink":
        toggle_subthinking()
        return True
    elif command == "/creative-mode":
        toggle_creative_mode()
        return True
    elif command == "/toggle-verify":
        toggle_auto_verification()
        return True
    elif command == "/toggle-trust":
        toggle_trust_mode()
        return True
    elif command.startswith("/analyze "):
        parts = command.split(maxsplit=1)
        if len(parts) == 2:
            file_path = parts[1]
            analyze_specific_file(file_path)
        else:
            console.print(Panel(
                "[bold red]Error:[/bold red] Missing file path. Use /analyze <file_path>", expand=False))
        return True
    elif command.startswith("/ststeps "):
        parts = command.split(maxsplit=1)
        if len(parts) == 2:
            set_subthinking_steps(parts[1])
        else:
            console.print(Panel(
                "[bold red]Error:[/bold red] Missing steps value. Use /ststeps <number>", expand=False))
        return True
    return False

def analyze_specific_file(file_path):
    """Analyze a specific file for code issues when requested via /analyze command."""
    if not os.path.exists(file_path):
        console.print(Panel(
            f"[bold red]Error:[/bold red] File '{escape(file_path)}' not found.", expand=False))
        return
        
    if not os.path.isfile(file_path):
        console.print(Panel(
            f"[bold red]Error:[/bold red] Path '{escape(file_path)}' is not a file.", expand=False))
        return
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Call the analyze_code function
        has_issues, result = analyze_code(file_path, content)
        
        if not has_issues:
            console.print(Panel(
                f"[bold green]Analysis complete:[/bold green] {result}", expand=False))
            
    except UnicodeDecodeError:
        console.print(Panel(
            f"[bold red]Error:[/bold red] Could not decode file '{escape(file_path)}' with utf-8. It might be a binary file.", expand=False))
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/bold red] An unexpected error occurred while analyzing '{escape(file_path)}'. Reason: {e}", expand=False))

def track_error_message(error_msg):
    """Tracks error messages and identifies if they relate to recently modified files."""
    global agent_context
    
    # Store the error message
    agent_context["reported_errors"].insert(0, error_msg)
    if len(agent_context["reported_errors"]) > 5:
        agent_context["reported_errors"].pop()
    
    # Check if this is a common file-related error
    file_path = None
    line_number = None
    
    # Try to extract file path and line number from common error patterns
    if "line" in error_msg and ":" in error_msg:
        # Look for patterns like "filename.py:10" or "Error at line 10"
        error_parts = error_msg.split()
        for i, part in enumerate(error_parts):
            if "line" in part and i+1 < len(error_parts) and error_parts[i+1].isdigit():
                line_number = int(error_parts[i+1])
            elif "line" in part and ":" in part:
                # Handle "line:10" format
                try:
                    line_number = int(part.split(":")[1])
                except (IndexError, ValueError):
                    pass
            
            # Check for file path patterns
            if ".py" in part:
                possible_path = part.strip(",:;\"'")
                if os.path.exists(possible_path):
                    file_path = possible_path
    
    # If we found a line number but no file path, check recently modified files
    if line_number and not file_path and agent_context["last_modified_files"]:
        file_path = agent_context["last_modified_files"][0]  # Most recently modified file
    
    # Return identified error details
    return {
        "error_message": error_msg,
        "file_path": file_path,
        "line_number": line_number,
        "context": {
            "last_modified_files": agent_context["last_modified_files"].copy(),
            "last_actions": agent_context["last_actions"][:3]  # Just the 3 most recent actions
        }
    }


def perform_subthinking(agent_executor, user_input, chat_history, steps=3):
    """Perform a series of subthinking steps before executing the agent."""
    global thinking_status, agent_context, creative_mode
    subthinking_results = []
    
    # Check if the user input contains error messages that need additional context
    error_context = None
    if any(error_term in user_input.lower() for error_term in ["error", "exception", "traceback", "failed", "issue", "bug", "problem", "crash", "line"]):
        error_context = track_error_message(user_input)
    
    # Get the LLM from setup_agent function directly
    try:
        # Access the LLM directly instead of through agent_executor
        # Set temperature based on creative mode
        temperature = 0.7 if creative_mode else 0
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17", 
            google_api_key=os.environ.get("GOOGLE_API_KEY"), 
            temperature=temperature
        )
    except Exception as e:
        console.print(Panel(f"[bold red]Error initializing LLM for subthinking: {e}[/bold red]", expand=False))
        raise e
    
    for i in range(1, steps + 1):
        console.print(Panel(f"[bold blue]Subthinking step {i}/{steps}...[/bold blue]", expand=False))
        
        # Create a modified prompt for subthinking
        if i == 1:
            context_info = ""
            if error_context:
                context_info = f"""
IMPORTANT CONTEXT: 
The user is reporting an error. Here's relevant information from the context:
- Recently modified files: {', '.join(error_context['context']['last_modified_files']) if error_context['context']['last_modified_files'] else 'None'}
- Recent actions: {str(error_context['context']['last_actions']) if error_context['context']['last_actions'] else 'None'}
- Possible file with error: {error_context['file_path'] if error_context['file_path'] else 'Unknown'}
- Possible line number with error: {error_context['line_number'] if error_context['line_number'] else 'Unknown'}

If the user is reporting a problem with a file you recently modified, be smart about it and check that file at the reported line number instead of asking for the file path.
"""
            elif agent_context["last_modified_files"] or agent_context["last_actions"]:
                context_info = f"""
IMPORTANT CONTEXT:
- Recently modified files: {', '.join(agent_context['last_modified_files']) if agent_context['last_modified_files'] else 'None'}
- Recent actions: {str(agent_context['last_actions'][:3]) if agent_context['last_actions'] else 'None'}

Consider this context when analyzing the request.
"""
                
            subthink_prompt = f"""SUBTHINKING STEP {i}/{steps}: You are in a planning phase called 'subthinking'. During this phase, you CANNOT use any tools - you are only planning.

{context_info}

IMPORTANT: For code modifications, prefer using search/replace (modify_file) over apply_diff as it's simpler and more direct.

Analyze the following request without taking action: What is being asked? What steps would you need to take? What tools would you need to use later? Identify any potential issues or ambiguities.

IMPORTANT: Try to DEEPLY understand what the user means, not just what they say. Consider:
1. What is the user's actual goal, beyond their literal request?
2. What would be the most helpful response that requires the fewest follow-up questions?
3. Can you make reasonable assumptions rather than asking questions?
4. If there are ambiguities, choose the most likely interpretation based on context.

USER REQUEST: {user_input}"""
        elif i == 2:
            subthink_prompt = f"""SUBTHINKING STEP {i}/{steps}: You are in a planning phase called 'subthinking'. During this phase, you CANNOT use any tools - you are only planning.

IMPORTANT: For code modifications, prefer using search/replace (modify_file) over apply_diff as it's simpler and more direct.

Based on your initial analysis, create a detailed plan of action with specific steps to address the request. List exactly which tools you would use and in what order. Consider edge cases and validations needed.

IMPORTANT: To avoid asking unnecessary questions:
1. Make reasonable assumptions when information is ambiguous
2. Favor taking action with sensible defaults over asking for clarification
3. If you need to make a decision between multiple paths, choose the most likely one based on context
4. Treat code as the primary interface - prefer to modify code rather than ask questions
5. Only ask questions when absolutely necessary.

Previous analysis:
{subthinking_results[0]}

USER REQUEST: {user_input}"""
        else:
            # For later steps, include previous subthinking
            previous_results = "\n\n".join([f"Step {j+1}: {result}" for j, result in enumerate(subthinking_results)])
            subthink_prompt = f"""SUBTHINKING STEP {i}/{steps}: You are in a planning phase called 'subthinking'. During this phase, you CANNOT use any tools - you are only planning.

IMPORTANT: For code modifications, prefer using search/replace (modify_file) over apply_diff as it's simpler and more direct.

Based on your previous analysis, refine your approach and prepare for execution. What specific tools will you call? What parameters will you use? What code will you need to modify? Be specific.

IMPORTANT:
1. Be decisive - make the best decisions based on available information
2. Avoid hesitation or asking unnecessary questions
3. Prioritize providing value over seeking clarification
4. If multiple valid approaches exist, choose one confidently rather than presenting options
5. Default to taking action rather than asking for confirmation
6. For most code edits, you should use the modify_file tool (search/replace) rather than apply_diff

Previous analysis:
{previous_results}

USER REQUEST: {user_input}"""
        
        # Use a regular status indicator rather than a Live display
        console.print(f"[bold yellow]Subthinking step {i}/{steps}...[/bold yellow]")
        
        try:
            # Make sure any existing status is stopped before creating a new one
            if thinking_status is not None:
                thinking_status.stop()
                thinking_status = None
                
            # Set up thinking status
            thinking_status = Status(f"Subthinking step {i}/{steps}...", spinner="dots", console=console)
            thinking_status.start()
            
            # Call the LLM directly without tools for subthinking
            messages = [
                SystemMessage(content="""You are in a subthinking phase where you CANNOT use any tools. You are only planning actions for later execution.
                
You are Coda2, an AI assistant that can perform tasks related to code development, file system management, and terminal operations.
You will eventually have access to tools for listing directory contents, reading files, creating files, writing to files, renaming files/directories, deleting files/directories, running terminal commands, and modifying specific text in files.

But first, you must plan carefully without using any tools.

IMPORTANT: BE SMART about error messages. If a user reports an error in a file you've recently modified, immediately check that file and focus on helping instead of asking for information you already have.
If a user reports an error in line X, look at that line and the surrounding code - don't ask which file it's in if you've recently modified a file.

IMPORTANT: REDUCE QUESTIONS. Always try to make educated guesses and take action rather than asking the user for more information. Only ask questions when absolutely necessary."""),
                HumanMessage(content=subthink_prompt)
            ]
            response = llm.invoke(messages)
            
            # Stop thinking status
            if thinking_status:
                thinking_status.stop()
                thinking_status = None
                
            subthink_output = response.content.strip()
            subthinking_results.append(subthink_output)
            
            # Display the subthinking output
            console.print(Panel(f"[bold magenta]Subthinking {i}/{steps}:[/bold magenta] {escape(subthink_output)}",
                            title=f"[bold magenta]Subthinking Step {i}/{steps}[/bold magenta]",
                            expand=False))
        except Exception as e:
            # Stop thinking status on error
            if thinking_status:
                thinking_status.stop()
                thinking_status = None
                
            console.print(Panel(f"[bold red]Error during subthinking step {i}/{steps}: {e}[/bold red]", expand=False))
            subthinking_results.append(f"Error: {str(e)}")
    
    # Prepare the final execution with subthinking context
    all_subthinking = "\n\n".join([f"Subthinking Step {i+1}/{steps}: {result}" for i, result in enumerate(subthinking_results)])
    
    # Add context information to the final prompt
    context_str = ""
    if error_context:
        context_str = f"""
IMPORTANT - PAY ATTENTION: 
The user reported an error. Here's relevant information from the context:
- Recently modified files: {', '.join(error_context['context']['last_modified_files']) if error_context['context']['last_modified_files'] else 'None'}
- Recent actions: {str(error_context['context']['last_actions']) if error_context['context']['last_actions'] else 'None'}
- Possible file with error: {error_context['file_path'] if error_context['file_path'] else 'Unknown'}
- Possible line number with error: {error_context['line_number'] if error_context['line_number'] else 'Unknown'}

Be smart and check the relevant files directly instead of asking for information you should already know!
"""
    elif agent_context["last_modified_files"] or agent_context["last_actions"]:
        context_str = f"""
IMPORTANT CONTEXT:
- Recently modified files: {', '.join(agent_context['last_modified_files']) if agent_context['last_modified_files'] else 'None'}
- Recent actions: {str(agent_context['last_actions'][:3]) if agent_context['last_actions'] else 'None'}

Remember this context as you execute the action.
"""
    
    final_prompt = f"""Now you can execute the request based on your subthinking analysis. You can now use tools to complete the request.

{context_str}

SUBTHINKING SUMMARY:
{all_subthinking}

ORIGINAL REQUEST: {user_input}

IMPORTANT: Minimize questions to the user. When faced with ambiguity:
1. Make educated guesses based on context
2. Prefer to take action with sensible defaults
3. Use existing patterns in code
4. Look at the code or context to answer your own questions
5. Only ask when truly necessary and the answer is critical

You have access to the following tools:
- list_directory - To see contents of directories
- read_file - To read file contents
- create_file - To create new files
- write_file - To overwrite existing files
- rename_path - To rename files or directories
- delete_path - To delete files or directories
- run_command - To execute terminal commands
- modify_file - To search and replace text in files"""
    
    console.print(Panel("[bold green]Executing request based on subthinking analysis...[/bold green]", expand=False))
    
    # Execute the final request
    console.print("[bold yellow]Executing final action...[/bold yellow]")
    
    try:
        # Add the subthinking results to chat history
        temp_history = chat_history.copy() if chat_history else []
        temp_history.append(SystemMessage(content=f"SUBTHINKING SUMMARY: {all_subthinking}"))
        
        # Make sure any existing status is stopped before creating a new one
        if thinking_status is not None:
            thinking_status.stop()
            thinking_status = None
            
        # Set up thinking status
        thinking_status = Status("Executing final action...", spinner="dots", console=console)
        thinking_status.start()
        
        # Call the agent with the final prompt (now with tools enabled)
        response = agent_executor.invoke({
            "input": final_prompt,
            "chat_history": temp_history  # Include subthinking context
        })
        
        # Stop thinking status
        if thinking_status:
            thinking_status.stop()
            thinking_status = None
            
        # Save history after executing a command
        save_history_to_file()
            
        return response
    except Exception as e:
        # Stop thinking status on error
        if thinking_status:
            thinking_status.stop()
            thinking_status = None
            
        console.print(Panel(f"[bold red]Error during final execution: {e}[/bold red]", expand=False))
        raise e

# --- Main CLI Logic ---


def main():
    # Removed session timer start time
    # session_start_time = time.time()

    # Declare thinking_status as global so it can be modified
    global thinking_status, subthinking_enabled, subthinking_steps, agent_context
    
    # Load history from file at startup
    load_history_from_file()
    
    console.print(Panel("Welcome to Coda2 Agent!",
                  title="[bold blue]Coda2[/bold blue]", expand=False))

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        console.print(Panel("[bold red]Error:[/bold red] GOOGLE_API_KEY not found in environment variables.\nPlease set the GOOGLE_API_KEY environment variable before running coda2.\n\nExample: export GOOGLE_API_KEY='YOUR_API_KEY'", expand=False))
        sys.exit(1)

    try:
        agent_executor = setup_agent(google_api_key)
        console.print(Panel(
            "[bold green]Agent Initialized:[/bold green] Gemini model and tools are ready.", expand=False))
    except Exception as e:
        error_message = f"[bold red]Error:[/bold red] Failed to initialize agent. Reason: {e}"
        console.print(Panel(
            error_message, title="[bold red]Initialization Error[/bold red]", expand=False))
        console.print(Panel(
            "Please ensure your GOOGLE_API_KEY is correct and the necessary libraries are installed.", expand=False))
        sys.exit(1)

    console.print(Panel(
        "Agent is ready. Type '[bold cyan]quit[/bold cyan]' to exit or '[bold cyan]/help[/bold cyan]' for available commands.", expand=False))

    # Initialize empty chat history
    chat_history = []

    # Initialize prompt_toolkit session with history and key bindings
    cmd_history_file = os.path.join(os.path.expanduser("~"), ".coda2", "cmd_history")
    os.makedirs(os.path.dirname(cmd_history_file), exist_ok=True)
    session = PromptSession(history=FileHistory(cmd_history_file))

    # Main interaction loop
    while True:
        console.print("\n")  # Add space before prompt area

        # Removed session timer display logic
        # elapsed_time = time.time() - session_start_time
        # hours, remainder = divmod(int(elapsed_time), 3600)
        # minutes, seconds = divmod(remainder, 60)
        # time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        # console.print(f"[dim]Session time: {time_str}[/dim]")

        # Get user input with prompt_toolkit (supports arrow keys)
        try:
            user_input = session.prompt(
                HTML("<green><b>Enter your command:</b></green> "))
        except EOFError:  # Handle Ctrl+D to exit
            console.print(Panel("Exiting Coda2. Goodbye!",
                          title="[bold blue]Coda2[/bold blue]", expand=False))
            break
        except KeyboardInterrupt:  # Handle Ctrl+C during input
            console.print("[yellow]Keyboard interrupt during input.[/yellow]")
            continue  # Skip the rest of the loop and show prompt again

        if user_input.lower() == 'quit':
            console.print(Panel("Exiting Coda2. Goodbye!",
                          title="[bold blue]Coda2[/bold blue]", expand=False))
            break

        if not user_input.strip():
            continue

        # Process special commands
        if user_input.startswith('/'):
            if process_special_command(user_input):
                continue

        console.print(
            Panel(f"[bold blue]User:[/bold blue] {user_input}", expand=False))

        # Add user input to chat history
        chat_history.append(HumanMessage(content=user_input))

        # Manage chat history to prevent context explosion
        chat_history = manage_chat_history(chat_history)

        # Check if the input is possibly an error report
        is_error_report = any(error_term in user_input.lower() for error_term in 
                             ["error", "exception", "traceback", "failed", "issue", "bug", 
                              "problem", "crash", "line", "broke", "fix", "not working"])

        # Remember what files have been modified in this session
        modified_files_before = set(agent_context["last_modified_files"])
        
        # Invoke the agent
        try:
            # Ensure any existing status is stopped before starting a new one
            if thinking_status is not None:
                thinking_status.stop()
                thinking_status = None
                
            # Use Status for the thinking spinner - store in our shared variable
            thinking_status = Status(
                "Agent thinking...", spinner="dots", console=console)
            thinking_status.start()

            try:
                # Check if subthinking is enabled
                if subthinking_enabled:
                    response = perform_subthinking(
                        agent_executor, user_input, chat_history, subthinking_steps)
                else:
                    # If the input might be an error report, add context about recent actions
                    if is_error_report:
                        error_context = track_error_message(user_input)
                        context_msg = ""
                        if error_context["file_path"] or agent_context["last_modified_files"]:
                            context_msg = f"""
IMPORTANT CONTEXT - The user may be reporting an error:
- Recently modified files: {', '.join(error_context['context']['last_modified_files']) if error_context['context']['last_modified_files'] else 'None'}
- Recent actions: {str(error_context['context']['last_actions']) if error_context['context']['last_actions'] else 'None'}
- Possible file with error: {error_context['file_path'] if error_context['file_path'] else 'Unknown'}
- Possible line number with error: {error_context['line_number'] if error_context['line_number'] else 'Unknown'}

IMPORTANT: Be smart and check the relevant files directly instead of asking for information you should already know!
"""
                            aug_user_input = f"{context_msg}\n\nUser request: {user_input}"
                            response = agent_executor.invoke({
                                "input": aug_user_input,
                                "chat_history": chat_history
                            })
                        else:
                            response = agent_executor.invoke({
                                "input": user_input,
                                "chat_history": chat_history
                            })
                    elif agent_context["last_modified_files"] or agent_context["last_actions"]:
                        # Add context about recent actions to help the agent be more aware
                        context_msg = f"""
CONTEXT:
- Recently modified files: {', '.join(agent_context['last_modified_files']) if agent_context['last_modified_files'] else 'None'}
- Recent actions: {str(agent_context['last_actions'][:3]) if agent_context['last_actions'] else 'None'}

Keep this context in mind when addressing the user request that follows.

User request: {user_input}
"""
                        response = agent_executor.invoke({
                            "input": context_msg,
                            "chat_history": chat_history
                        })
                    else:
                        response = agent_executor.invoke({
                            "input": user_input,
                            "chat_history": chat_history
                        })

                # Stop the thinking status
                if thinking_status:
                    thinking_status.stop()
                thinking_status = None
            except Exception as e:
                # Make sure to stop the thinking status even on error
                if thinking_status:
                    thinking_status.stop()
                thinking_status = None
                raise e  # Re-raise to be caught by the outer exception handler

            # Get the final output from the response
            agent_output = response.get('output', '').strip()

            # Add agent response to chat history and print it
            if agent_output:
                chat_history.append(AIMessage(content=agent_output))
                console.print(Panel(f"[bold magenta]Agent:[/bold magenta] {escape(agent_output)}",
                                    title="[bold magenta]Agent Response[/bold magenta]",
                                    expand=False))
            
            # After agent execution, run code verification if enabled
            if auto_verify_enabled:
                # Find which files were modified during this execution
                modified_files_after = set(agent_context["last_modified_files"])
                newly_modified_files = modified_files_after - modified_files_before
                
                if newly_modified_files:
                    console.print(Panel("[bold blue]Starting automatic code verification...[/bold blue]",
                                  title="[bold blue]Code Verification[/bold blue]", expand=False))
                    
                    # Run verification on each modified file
                    issues_found = False
                    for file_path in newly_modified_files:
                        console.print(f"[dim]Verifying file: {escape(file_path)}[/dim]")
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            # Use the analyze_code function we created earlier
                            has_issues, result = analyze_code(file_path, content)
                            
                            if has_issues:
                                issues_found = True
                        except Exception as e:
                            console.print(f"[bold yellow]Warning: Could not verify file {escape(file_path)}: {e}[/bold yellow]")
                    
                    if not issues_found:
                        console.print(Panel("[bold green]Verification complete. No issues were found in the modified files.[/bold green]",
                                      expand=False))

        # FIX: Add specific handling for KeyboardInterrupt during agent execution
        except KeyboardInterrupt:
            console.print(Panel("[yellow]Agent execution interrupted by user.[/yellow]",
                                title="[yellow]Interrupted[/yellow]",
                                expand=False))
            # Ensure the thinking status is stopped if it's still active
            if thinking_status:
                thinking_status.stop()
            thinking_status = None
            continue  # Go back to the start of the loop for the next command
        except Exception as e:
            # Handle other errors without using Text.from_markup to avoid potential markup errors
            # Also escape the error message itself
            error_msg = f"An error occurred during agent execution: {escape(str(e))}"
            console.print(Panel(f"[bold red]{error_msg}[/bold red]",
                                title="[bold red]Agent Execution Error[/bold red]",
                                expand=False))
            # Ensure the thinking status is stopped if it's still active
            if thinking_status:
                thinking_status.stop()
            thinking_status = None
            # Add error to context
            track_error_message(str(e))
            # Optionally add error to chat history if needed
            # chat_history.append(AIMessage(content=f"Error during execution: {escape(str(e))}"))

        # After each successful command execution or any command processing
        save_history_to_file()

        # If we have accumulated a lot of messages, summarize older ones to save context
        if len(chat_history) > 8:  # Keep the most recent 8 messages
            chat_history = summarize_chat_history(chat_history, 8)


if __name__ == "__main__":
    main()
