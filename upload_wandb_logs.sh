#!/bin/bash

# Function to extract WandB API key from config files
get_wandb_key() {
    # Try reading from ~/.netrc first
    if [ -f ~/.netrc ]; then
        WANDB_KEY=$(grep -A2 'api.wandb.ai' ~/.netrc | grep password | awk '{print $2}')
        if [ ! -z "$WANDB_KEY" ]; then
            return
        fi
    fi
    
    # Try reading from wandb settings file
    WANDB_CONFIG=~/.config/wandb/settings
    if [ -f "$WANDB_CONFIG" ]; then
        WANDB_KEY=$(grep api_key "$WANDB_CONFIG" | cut -d: -f2 | tr -d ' "')
    fi
}

# Get the WandB API key
get_wandb_key

# Check if we found a valid API key
if [ -z "$WANDB_KEY" ]; then
    echo "Error: Could not find WandB API key in ~/.netrc or ~/.config/wandb/settings"
    echo "Please either:"
    echo "1. Log in using 'wandb login'"
    echo "2. Set WANDB_API_KEY environment variable"
    exit 1
fi

# Set the API key
export WANDB_API_KEY=$WANDB_KEY

# Default project name and sync interval
PROJECT_NAME="your_project_name"
SYNC_INTERVAL=5  # seconds

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --interval)
            SYNC_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--project project_name] [--interval seconds]"
            exit 1
            ;;
    esac
done

echo "Starting continuous upload of offline runs to project: $PROJECT_NAME"
echo "Sync interval: ${SYNC_INTERVAL} seconds"

# Track already synced runs to avoid re-syncing
declare -A synced_runs

# Continuous monitoring loop
while true; do
    # Find all offline run directories
    for run_dir in wandb/*offline-run-*; do
        if [ -d "$run_dir" ] && [ -z "${synced_runs[$run_dir]}" ]; then
            echo "Uploading new run: $run_dir"
            WANDB_PROJECT=$PROJECT_NAME wandb sync "$run_dir"
            synced_runs[$run_dir]=1
        fi
    done
    
    sleep $SYNC_INTERVAL
done