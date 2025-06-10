import os
import subprocess
import sys
from datetime import datetime

import openai
from dotenv import load_dotenv
from github import Github

# Load environment variables
load_dotenv()

# Get environment variables from GitHub Actions
PR_NUMBER = os.getenv("PR_NUMBER")
REPO_NAME = os.getenv("REPO_NAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PR_TITLE = os.getenv("PR_TITLE")
PR_BODY = os.getenv("PR_BODY")

# Template for generating commit message
generate_commit_message_template = """
I have a pull request with the following details:

Title: {pr_title}
Description: {pr_body}

Commit messages from the PR:
{commit_messages}

Generate a concise commit message that summarizes the changes made in this pull request.

Follow these guidelines:
1. Use imperative mood (e.g., "Add feature" not "Added feature")
2. Keep the first line under 50 characters
3. If needed, add a longer description after a blank line
4. Focus on the "what" and "why" of the changes
5. Use semantic prefixes like feat:, fix:, docs:, refactor:, etc.
6. Be specific and clear

Return ONLY the commit message, nothing else.

Example format:
feat: Add user authentication system

Implement OAuth2-based authentication with JWT tokens.
Includes login, logout, and session management.

Closes #{pr_number}
"""


def get_pr_commits(repo_name, pr_number, github_token):
    """Get commits from a specific PR"""
    if not github_token:
        raise RuntimeError("Please set GITHUB_TOKEN in your environment")

    gh = Github(github_token)
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(int(pr_number))

    commit_messages = []
    for commit in pr.get_commits():
        commit_messages.append(commit.commit.message)

    return commit_messages


def generate_commit_message(pr_title, pr_body, commit_messages, pr_number):
    """Use OpenAI to generate a summarized commit message"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Call the OpenAI API to generate the commit message
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,  # Lower temperature for more consistent results
        messages=[
            {
                "role": "system",
                "content": "You are an expert software developer who writes excellent Git commit messages following best practices and conventional commit standards.",
            },
            {
                "role": "user",
                "content": generate_commit_message_template.format(
                    pr_title=pr_title or "No title provided",
                    pr_body=pr_body or "No description provided",
                    commit_messages=(
                        "\n".join(commit_messages)
                        if commit_messages
                        else "No commit messages found"
                    ),
                    pr_number=pr_number,
                ),
            },
        ],
    )

    return completion.choices[0].message.content.strip()


def get_latest_commit_hash():
    """Get the hash of the latest commit on current branch"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting latest commit hash: {e}")
        return None


def update_commit_message(new_message):
    """Update the commit message of the latest commit"""
    try:
        # Amend the commit message
        subprocess.run(["git", "commit", "--amend", "-m", new_message], check=True)
        print("Successfully updated commit message")

        # Force push the updated commit
        subprocess.run(
            ["git", "push", "--force-with-lease", "origin", "main"], check=True
        )
        print("Successfully pushed updated commit to main")

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error updating commit message: {e}")
        return False


def is_merge_commit():
    """Check if the latest commit is a merge commit"""
    try:
        result = subprocess.run(
            ["git", "cat-file", "-p", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_info = result.stdout
        # Merge commits have two or more parent lines
        parent_count = commit_info.count("parent ")
        return parent_count >= 2
    except subprocess.CalledProcessError:
        return False


def main():
    if not all([PR_NUMBER, REPO_NAME, GITHUB_TOKEN, OPENAI_API_KEY]):
        missing = []
        if not PR_NUMBER:
            missing.append("PR_NUMBER")
        if not REPO_NAME:
            missing.append("REPO_NAME")
        if not GITHUB_TOKEN:
            missing.append("GITHUB_TOKEN")
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        print(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    # Check if the latest commit is a merge commit
    if not is_merge_commit():
        print("Latest commit is not a merge commit, skipping...")
        return

    # Get commits from the PR
    commit_messages = get_pr_commits(REPO_NAME, PR_NUMBER, GITHUB_TOKEN)

    # Generate the new commit message
    new_commit_message = generate_commit_message(
        PR_TITLE, PR_BODY, commit_messages, PR_NUMBER
    )

    print(f"Generated commit message:\n{new_commit_message}")

    # Get current commit hash for logging
    current_hash = get_latest_commit_hash()
    if current_hash:
        print(f"Updating commit: {current_hash}")

    # Update the commit message
    if update_commit_message(new_commit_message):
        print(f"Successfully updated merge commit message for PR #{PR_NUMBER}")
    else:
        print(f"Failed to update merge commit message for PR #{PR_NUMBER}")
        sys.exit(1)


if __name__ == "__main__":
    main()
