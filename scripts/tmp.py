import os
import sys
import subprocess


from dotenv import load_dotenv
from datetime import datetime

import requests

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME")
TAG = os.getenv('TAG')

def run_git_command(command):
    try:
        result = subprocess.run(['git'] + command.split(), 
                                check=True, 
                                capture_output=True, 
                                text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return e.stderr
    
def bash_command(command):
    try:
        result = subprocess.run(command.split(), 
                                check=True, 
                                capture_output=True, 
                                text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return e.stderr

def generate_changelog_message():
    # Example usage:
    bash_command('git fetch --all --tags')
    current_version = bash_command('git tag -l --sort=v:refname').strip()
    print(f"Current Version: {current_version}")
    print(f'git log --pretty=format:"%h-%s" {current_version}..')


headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def get_release_ids():
    

    api_url = f"https://api.github.com/repos/{REPO_NAME}/releases"
    print(api_url)
    response = requests.get(api_url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        release_data = response.json()
        print("Release Data:")
        for release in release_data:
            print(f"Release ID: {release['id']}, Tag Name: {release['tag_name']}")
    else:
        print(f"Failed to fetch release data: {response.status_code}")
        print(response.json())

        
if __name__ == "__main__":

    get_release_ids()
    # generate_changelog_message()