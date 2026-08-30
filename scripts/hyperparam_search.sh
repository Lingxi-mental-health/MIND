#!/bin/bash

# Default parameters
ENV_NAME=${ENV_NAME:-"sokoban"}
EXP_BASE_NAME=${EXP_BASE_NAME:-"sokoban"}
DRY_RUN=${DRY_RUN:-""}

# Parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --env_name=*)
      ENV_NAME="${1#*=}"
      ;;
    --exp_base_name=*)
      EXP_BASE_NAME="${1#*=}"
      ;;
    --dry_run)
      DRY_RUN="--dry_run"
      ;;
    --search_group)
      SEARCH_GROUP="$2"
      shift
      ;;
    *)
      # Pass through any other arguments
      EXTRA_ARGS="$EXTRA_ARGS $1"
      ;;
  esac
  shift
done

mkdir -p ./log/terminal

# Print the parameters being used
echo "Using parameters:"
echo "Environment: $ENV_NAME"
echo "Experiment base name: $EXP_BASE_NAME"
echo "Dry run: ${DRY_RUN:-false}"
echo "Search group: ${SEARCH_GROUP}"
echo "-------------------"

# Execute the parameter search script
python scripts/hyperparam_search.py \
  --env_name "$ENV_NAME" \
  --base_experiment_name "$EXP_BASE_NAME" \
  --search_group "$SEARCH_GROUP" \
  $DRY_RUN \
  $EXTRA_ARGS \
  >> ./log/terminal/hyperparam_search_${EXP_BASE_NAME}_$(date +%Y%m%d_%H%M%S).log