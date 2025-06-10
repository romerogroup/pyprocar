import os
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
EVENT_TYPE = os.getenv("EVENT_TYPE")

# Template for summarizing commits
summarize_commit_message_template = """
I have the following commit messages:

{commit_messages}

{previous_summary_section}

Classify them into the following categories:
1. Bugs
2. New Features
3. Documentation
4. Maintenance

Then, summarize the changes for each category into a bullet point list. 

Follow these instructions:
1. Use Markdown formatting
2. Use 5 "#" to indicate a new category.
3. Do not number the categories.
4. If there are no changes, return "No changes"
5. Improve the writing from the commit message and be concise.
6. RETURN ONLY THE SUMMARY IN BULLET POINT FORMAT.
{previous_summary_instruction}

Here is an example:

##### Bugs
- None identified
##### New features
- Added new action to execute python script on push to main
##### Documentation updates
- Updated readme
##### Maintenance
- Moved GitHub actions scripts directory to root
- Added tests
- Changed version
"""

# Template for generating PR merge message
generate_merge_message_template = """
I have a pull request with the following details:

Title: {pr_title}
Description: {pr_body}

Commit messages from the PR:
{commit_messages}

Summary of changes by category:
{summary}

Generate a concise merge commit message that summarizes the changes made in this pull request.
The git message should focus on the intent of the changes, not the technical details.
After describing the intent, it should focus on a high level summary of the changes.

Use the summary above to understand what categories of changes were made and incorporate that understanding into the merge message.

Follow these formatting rules:
1. Use imperative mood (e.g., "Add feature" not "Added feature")
2. Keep the first line under 72 characters
3. If needed, add a longer description after a blank line
4. Use semantic prefixes like feat:, fix:, docs:, refactor:, etc.
5. Use the present tense
6. Be specific and clear, but concise

Example:

feat: Add user authentication and session management

This commit introduces a complete user authentication flow, enabling users to sign up, log in, and log out. The primary goal is to secure user-specific data and create personalized experiences.

- Implement email/password login and registration forms.
- Add server-side logic for validating credentials and creating user sessions.
- Introduce a new `auth` module to handle all authentication-related logic.
- Create protected routes that require a valid user session to access.

Return ONLY the merge commit message, nothing else.
"""

# Marker to identify our bot comments
COMMENT_MARKER = "<!-- PR_SUMMARY_BOT_COMMENT -->"


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


def find_existing_summary_comment(repo_name, pr_number, github_token):
    """Find existing summary comment created by this bot and return its content"""
    if not github_token:
        raise RuntimeError("Please set GITHUB_TOKEN in your environment")

    gh = Github(github_token)
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(int(pr_number))

    # Look for existing comments with our marker
    for comment in pr.get_issue_comments():
        if COMMENT_MARKER in comment.body:
            return comment, extract_summary_from_comment(comment.body)

    return None, None


def extract_summary_from_comment(comment_body):
    """Extract the summary content from a bot comment"""
    lines = comment_body.split("\n")
    summary_lines = []
    in_summary = False

    for line in lines:
        # Start capturing after the header and date
        if line.startswith("*Last updated:"):
            in_summary = True
            continue
        # Stop capturing at the footer
        if line.startswith("---") and in_summary:
            break
        # Capture summary content
        if in_summary and line.strip():
            summary_lines.append(line)

    return "\n".join(summary_lines).strip()


def summarize_commit_messages(commit_messages, previous_summary=None):
    """Use OpenAI to summarize commit messages"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Prepare previous summary section
    if previous_summary:
        previous_summary_section = f"""
Previous summary from this PR:
{previous_summary}

Please update the summary above to include the new commit messages below, maintaining consistency and building upon the previous analysis.
"""
        previous_summary_instruction = "\n7. If a previous summary exists, update it to include new changes while maintaining consistency."
    else:
        previous_summary_section = ""
        previous_summary_instruction = ""

    # Call the OpenAI API to classify and summarize the changes
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1.0,
        messages=[
            {
                "role": "system",
                "content": "You are a professional programmer that is excellent with github repository management. Your goal is to help the user with tasks\n",
            },
            {
                "role": "user",
                "content": summarize_commit_message_template.format(
                    commit_messages="\n".join(commit_messages),
                    previous_summary_section=previous_summary_section,
                    previous_summary_instruction=previous_summary_instruction,
                ),
            },
        ],
    )

    changes_summary = completion.choices[0].message.content.strip()
    return changes_summary


def get_pr_details(repo_name, pr_number, github_token):
    """Get PR title and body"""
    if not github_token:
        raise RuntimeError("Please set GITHUB_TOKEN in your environment")

    gh = Github(github_token)
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(int(pr_number))

    return pr.title, pr.body or ""


def generate_merge_message(pr_title, pr_body, commit_messages, summary):
    """Use OpenAI to generate a merge commit message"""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Call the OpenAI API to generate the merge message
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
                "content": generate_merge_message_template.format(
                    pr_title=pr_title or "No title provided",
                    pr_body=pr_body or "No description provided",
                    commit_messages=(
                        "\n".join(commit_messages)
                        if commit_messages
                        else "No commit messages found"
                    ),
                    summary=summary or "No summary available",
                ),
            },
        ],
    )

    return completion.choices[0].message.content.strip()


def create_or_update_summary_comment(
    repo_name, pr_number, summary, github_token, event_type, merge_message=None
):
    """Create a new summary comment or update existing one"""
    if not github_token:
        raise RuntimeError("Please set GITHUB_TOKEN in your environment")

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add merge message section if provided
    merge_section = ""
    if merge_message:
        merge_section = f"""

## 🚀 Suggested Merge Commit Message

```
{merge_message}
```

"""

    # Create comment body with marker
    comment_body = f"""{COMMENT_MARKER}
## 📋 Pull Request Summary

*Last updated: {current_date} (Event: {event_type})*

{summary}{merge_section}

---
*This comment was automatically generated by the PR summary workflow.*
"""

    existing_comment, _ = find_existing_summary_comment(
        repo_name, pr_number, github_token
    )

    if existing_comment:
        # Update existing comment
        existing_comment.edit(comment_body)
        print(f"Updated existing summary comment on PR #{pr_number}")
    else:
        # Create new comment
        gh = Github(github_token)
        repo = gh.get_repo(repo_name)
        pr = repo.get_pull(int(pr_number))
        pr.create_issue_comment(comment_body)
        print(f"Created new summary comment on PR #{pr_number}")


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

    # Get commits from the PR
    commit_messages = get_pr_commits(REPO_NAME, PR_NUMBER, GITHUB_TOKEN)

    if not commit_messages:
        print("No commits found in this PR")
        return

    # Check for existing summary
    _, previous_summary = find_existing_summary_comment(
        REPO_NAME, PR_NUMBER, GITHUB_TOKEN
    )

    if previous_summary:
        print("Found previous summary, including it in the prompt for context")
    else:
        print("No previous summary found, generating fresh summary")

    # Summarize the commits (with previous summary if available)
    summary = summarize_commit_messages(commit_messages, previous_summary)

    # Generate merge message on synchronize events
    merge_message = None
    pr_title, pr_body = get_pr_details(REPO_NAME, PR_NUMBER, GITHUB_TOKEN)
    merge_message = generate_merge_message(pr_title, pr_body, commit_messages, summary)

    # Create or update the summary comment
    create_or_update_summary_comment(
        REPO_NAME,
        PR_NUMBER,
        summary,
        GITHUB_TOKEN,
        EVENT_TYPE or "unknown",
        merge_message,
    )


if __name__ == "__main__":
    main()
