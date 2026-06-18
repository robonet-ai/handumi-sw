#!/usr/bin/env python3

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from dotenv import load_dotenv


DATASET_PATH = "/home/leonardo/NONHUMAN/dexumi/datasets/dexumi_demo"
REPO_ID = "NONHUMAN-RESEARCH/dexumi-dataset"
PRIVATE = False
COMMIT_MESSAGE = "Upload dataset"
BRANCH = "main"

load_dotenv()

def validate_dataset_structure(dataset_path: Path) -> bool:
    """
    Validates that the dataset has the required LeRobot structure.

    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        True if the structure is valid, False otherwise
    """
    required_dirs = ["data", "meta", "videos"]
    required_files = ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]
    
    print(f"Validating dataset structure at: {dataset_path}")
    
    # Check if directory exists
    if not dataset_path.exists():
        print(f"Error: Directory {dataset_path} does not exist")
        return False
    
    # Check for required directories
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            print(f"Warning: Directory '{dir_name}' not found")
        else:
            print(f"Found directory '{dir_name}'")
    
    # Check for required files
    for file_path in required_files:
        full_path = dataset_path / file_path
        if not full_path.exists():
            print(f"Warning: File '{file_path}' not found")
        else:
            print(f"Found file '{file_path}'")
    
    return True


def upload_dataset_to_huggingface() -> bool:
    """
    Uploads a local dataset to Hugging Face Hub using the script settings.
    
    Returns:
        True if upload was successful, False otherwise
    """
    # Get absolute path to dataset
    script_dir = Path(__file__).parent.parent  # Go to project root
    dataset_path = script_dir / DATASET_PATH
    
    # Validate dataset structure
    if not validate_dataset_structure(dataset_path):
        print("Warning: The dataset does not have the expected structure, continuing with upload...")
    
    try:
        # Get token from .env file
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("Error: HF_TOKEN not found in .env file")
            print("\nTo set up your token:")
            print("   1. Create a .env file in the project root")
            print("   2. Add: HF_TOKEN=your_token_here")
            print("   3. Get your token at: https://huggingface.co/settings/tokens")
            return False
        
        print("Token found in .env file")
        
        # Initialize API
        api = HfApi(token=token)
        
        # Verify authentication
        try:
            user_info = api.whoami()
            print(f"Authenticated as: {user_info['name']}")
        except Exception as e:
            print(f"Authentication error: {e}")
            print("   Make sure your HF_TOKEN in .env is valid")
            return False
        
        # Create repo if it does not exist
        print(f"\nCreating/verifying repo: {REPO_ID}")
        try:
            create_repo(
                repo_id=REPO_ID,
                repo_type="dataset",
                private=PRIVATE,
                exist_ok=True,
                token=token
            )
            print(f"Repo '{REPO_ID}' is ready")
        except Exception as e:
            print(f"Error creating repo: {e}")
            return False
        
        # Create branch if it doesn't exist (and it's not 'main')
        if BRANCH != "main":
            print(f"\nChecking/creating branch: {BRANCH}")
            try:
                # Try to get the branch
                api.list_repo_refs(repo_id=REPO_ID, repo_type="dataset")
                # Try to create the branch from main
                try:
                    api.create_branch(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        branch=BRANCH,
                        token=token
                    )
                    print(f"Branch '{BRANCH}' created successfully")
                except Exception as branch_error:
                    # Branch might already exist, which is fine
                    if "already exists" in str(branch_error).lower() or "reference already exists" in str(branch_error).lower():
                        print(f"Branch '{BRANCH}' already exists")
                    else:
                        print(f"Note: Could not create branch (might already exist): {branch_error}")
            except Exception as e:
                print(f"Warning checking branches: {e}")
        
        # Lista de branches a las que subir
        branches_to_upload = [BRANCH]
        if BRANCH != "main":
            branches_to_upload.append("main")
        
        # Upload the dataset to each branch
        for branch in branches_to_upload:
            print(f"\n{'='*60}")
            print(f"UPLOADING TO BRANCH: {branch}")
            print(f"{'='*60}")
            print(f"\nUploading dataset from {dataset_path}...")
            print(f"   Destination: https://huggingface.co/datasets/{REPO_ID}")
            print(f"   Branch: {branch}")
            print("   (This may take several minutes depending on size...)")
            print("   Note: Uploading ALL files without any filters")
            
            url = upload_folder(
                folder_path=str(dataset_path),
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=COMMIT_MESSAGE,
                revision=branch,
                token=token,
                allow_patterns=None,  # No filtering - upload everything
                ignore_patterns=None,  # Don't ignore anything
            )
            
            print(f"\n✓ Dataset uploaded successfully to branch '{branch}'!")
            print(f"   URL: {url}")
        
        print(f"\n{'='*60}")
        print(f"ALL UPLOADS COMPLETED")
        print(f"{'='*60}")
        print(f"View at: https://huggingface.co/datasets/{REPO_ID}")
        
        return True
        
    except Exception as e:
        print(f"\nError during upload: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("UPLOADING DATASET TO HUGGING FACE HUB")
    print("=" * 60)
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Repository: {REPO_ID}")
    print(f"Branch: {BRANCH}")
    print(f"Private: {'Yes' if PRIVATE else 'No'}")
    print()
    
    # Run upload
    success = upload_dataset_to_huggingface()
    
    if success:
        print("\n" + "=" * 60)
        print("PROCESS COMPLETED SUCCESSFULLY")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("PROCESS FAILED")
        print("=" * 60)
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
