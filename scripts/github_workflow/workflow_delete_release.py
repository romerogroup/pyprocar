import os
import requests

# Your GitHub token
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_NAME = os.getenv('GITHUB_REPOSITORY')
RELEASE_ID = os.getenv('RELEASE_ID')  # Pass the release ID from the GitHub Action context
TAG = os.getenv('TAG')

# GitHub API base URL for releases
api_base_url = f"https://api.github.com/repos/{REPO_NAME}"

# Headers for authentication
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

# Step 1: Delete the release using the release ID
def delete_release(release_id):
    if release_id:
        api_url = f"{api_base_url}/releases/{release_id}"
        response = requests.delete(api_url, headers=headers)
        if response.status_code == 204:
            print(f"Release with ID {release_id} deleted successfully.")
        else:
            print(f"Failed to delete release: {response.status_code}")
            print(response.json())

# Step 2: Optionally delete the tag associated with the release
def delete_tag(tag_name):
    api_url = f"{api_base_url}/git/refs/tags/{tag_name}"
    response = requests.delete(api_url, headers=headers)
    if response.status_code == 204:
        print(f"Tag {tag_name} deleted successfully.")
    else:
        print(f"Failed to delete tag {tag_name}: {response.status_code}")
        print(response.json())

# Main logic
if __name__ == "__main__":

    # Delete the release
    delete_release(RELEASE_ID)
    
    # Optionally delete the tag
    delete_tag(TAG)
