
import os
import subprocess
import sys
import requests

def get_releases_data(repo_name=None,github_token=None, verbose=False):
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    api_url = f"https://api.github.com/repos/{repo_name}/releases"
    if verbose:
        print(api_url)

    response = requests.get(api_url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        release_data = response.json()
        if verbose:
            print("Release Data:")
            for release in release_data:
                print(f"Release ID: {release['id']}, Tag Name: {release['tag_name']}")
    else:
        if verbose:
            print(f"Failed to fetch release data: {response.status_code}")
            print(response.json())

    return release_data

def run_git_command(command):
    try:
        result = subprocess.run(['git'] + command.split(), 
                                check=True, 
                                capture_output=True, 
                                text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        # print(f"An error occurred: {e}")
        return e.stderr
    
def bash_command(command):
    try:
        result = subprocess.run(command.split(), 
                                check=True, 
                                capture_output=True, 
                                text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        # print(f"An error occurred: {e}")
        return e.stderr