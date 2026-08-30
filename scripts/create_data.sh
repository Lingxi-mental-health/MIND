#!/bin/bash

# Exit on error
set -e

# Fix next seed generation by hash()
export PYTHONHASHSEED=10000

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Print warning with color
print_warning() {
    echo -e "${YELLOW}[Warning] ${1}${NC}"
}

# Create Sokoban dataset
create_sokoban_dataset() {
    print_step "Configuring Sokoban environment settings..."
        
    # Sokoban environment settings
    export DIM_X=6
    export DIM_Y=6
    export NUM_BOXES=1

    # generate normal difficulty dataset
    export MAX_STEPS=10
    export SEARCH_DEPTH=30
    
    print_step "Creating Sokoban dataset..."
    print_warning "Note: SOKOBAN errors during creation are normal and can be ignored"
    
    python ragen/env/sokoban/create_dataset.py \
        --output data/sokoban \
        --seed 10000 \
        --train_size 100000 \
        --test_size 500 \
        --prefix qwen-instruct
        
    echo -e "${GREEN}Sokoban dataset created successfully!${NC}"
}

# Create FrozenLake dataset
create_frozen_lake_dataset() {
    print_step "Configuring FrozenLake environment settings..."
    
    # FrozenLake environment settings
    export SIZE=4  # size * size grid
    export P=0.8   # percentage of frozen tiles
    
    print_step "Creating FrozenLake dataset..."
    
    python ragen/env/frozen_lake/create_dataset.py \
        --output data/frozenlake \
        --seed 100000 \
        --train_size 100000 \
        --test_size 500 \
        --prefix qwen-instruct
        
    echo -e "${GREEN}FrozenLake dataset created successfully!${NC}"
}

# Create Two-Armed Bandit dataset
create_two_armed_bandit_dataset() {
    print_step "Configuring Two-Armed Bandit environment settings..."
    
    # Two-Armed Bandit environment settings
    export LOW_RISK_NAME=phoenix
    export HIGH_RISK_NAME=dragon
    
    print_step "Creating Two-Armed Bandit dataset..."
    
    python ragen/env/bandit/create_dataset_two_armed.py \
        --output data/two_armed_bandit \
        --seed 100000 \
        --train_size 100000 \
        --test_size 500 \
        --prefix qwen-instruct
        
    echo -e "${GREEN}Two-Armed Bandit dataset created successfully!${NC}"
}


# Main function
main() {
    # Create data directory if it doesn't exist
    mkdir -p data/sokoban data/frozenlake data/two_armed_bandit
    
    # Create datasets
    create_sokoban_dataset
    create_frozen_lake_dataset
    create_two_armed_bandit_dataset
    
    echo -e "${GREEN}All datasets created successfully!${NC}"
}

# Run main function
main