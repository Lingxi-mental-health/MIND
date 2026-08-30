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

# Create Sokoban dataset with specified parameters
create_sokoban_dataset() {
    local name=$1
    local dim_x=$2
    local dim_y=$3
    local num_boxes=$4
    local search_depth=$5
    
    print_step "Configuring ${name} environment settings..."
    
    # Sokoban environment settings
    export DIM_X=$dim_x
    export DIM_Y=$dim_y
    export NUM_BOXES=$num_boxes
    export MAX_STEPS=10
    export SEARCH_DEPTH=$search_depth
    
    print_step "Creating Sokoban ${name} dataset..."
    print_warning "Note: SOKOBAN errors during creation are normal and can be ignored"
    
    python ragen/env/sokoban/create_dataset.py \
        --output "data/${name}" \
        --seed 10000 \
        --train_size 10000 \
        --test_size 10 \
        --prefix qwen-instruct
        
    echo -e "${GREEN}Sokoban ${name} dataset created successfully!${NC}"
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
        --train_size 10000 \
        --test_size 10 \
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
        --train_size 10000 \
        --test_size 10 \
        --prefix qwen-instruct
        
    echo -e "${GREEN}Two-Armed Bandit dataset created successfully!${NC}"
}

# Main function
main() {
    # Create all data directories
    mkdir -p data/sokoban data/sokoban_hard data/sokoban_xhard \
            data/sokoban_large data/sokoban_xlarge \
            data/sokoban_multi data/sokoban_xmulti \
            data/frozenlake \
            data/two_armed_bandit
    
    # Create normal Sokoban dataset (baseline)
    create_sokoban_dataset "sokoban" 6 6 1 30
    
    # Create hard difficulty datasets
    create_sokoban_dataset "sokoban_hard" 6 6 1 100
    create_sokoban_dataset "sokoban_xhard" 6 6 1 500
    
    # Create larger grid datasets
    create_sokoban_dataset "sokoban_large" 8 8 1 30
    create_sokoban_dataset "sokoban_xlarge" 10 10 1 30
    
    # Create multi-box datasets
    create_sokoban_dataset "sokoban_multi" 6 6 2 30
    
    # Create FrozenLake dataset
    create_frozen_lake_dataset
    
    # Create Two-Armed Bandit dataset
    create_two_armed_bandit_dataset
    
    echo -e "${GREEN}All datasets created successfully!${NC}"
}

# Run main function
main