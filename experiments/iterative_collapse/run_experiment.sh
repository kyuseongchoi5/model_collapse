#!/bin/bash
# Convenience script for running model collapse experiments

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
EXPERIMENT_TYPE="quick_test"
DEVICE="auto"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [EXPERIMENT_TYPE] [OPTIONS]

EXPERIMENT_TYPES:
    quick_test          Quick test run (20 epochs, small model)
    full                Full experiment (200 epochs, default model)
    early_switch        Switch at epoch 50
    late_switch         Switch at epoch 150
    mean_mode           Use mean predictions (faster collapse)
    sample_mode         Use sampling (default, slower collapse)
    mixed_data          50% real + 50% synthetic
    large_model         Large model experiment
    custom              Custom parameters (specify via options)

OPTIONS:
    --switch_epoch N    Epoch to switch data source
    --total_epochs N    Total training epochs
    --batch_size N      Batch size
    --device DEVICE     'cpu', 'cuda', or 'auto' (default: auto)
    --help              Show this help message

EXAMPLES:
    # Quick test
    $0 quick_test

    # Full experiment
    $0 full

    # Custom configuration
    $0 custom --switch_epoch 75 --total_epochs 150

    # Early switch with specific device
    $0 early_switch --device cuda:0

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        quick_test|full|early_switch|late_switch|mean_mode|sample_mode|mixed_data|large_model|custom)
            EXPERIMENT_TYPE="$1"
            shift
            ;;
        --switch_epoch)
            SWITCH_EPOCH="$2"
            shift 2
            ;;
        --total_epochs)
            TOTAL_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

# Set up experiment-specific parameters
case $EXPERIMENT_TYPE in
    quick_test)
        print_info "Running quick test experiment"
        EXP_NAME="quick_test"
        ARGS="--experiment_name quick_test \
              --switch_epoch 10 \
              --total_epochs 20 \
              --batch_size 200 \
              --steps_per_epoch 10 \
              --emsize 256 \
              --nlayers 3 \
              --nhid 512 \
              --nhead 4"
        ;;

    full)
        print_info "Running full experiment"
        EXP_NAME="full_experiment"
        ARGS="--experiment_name full_experiment \
              --switch_epoch 100 \
              --total_epochs 200 \
              --config config.yaml"
        ;;

    early_switch)
        print_info "Running early switch experiment (epoch 50)"
        EXP_NAME="early_switch_50"
        ARGS="--experiment_name early_switch_50 \
              --switch_epoch 50 \
              --total_epochs 200 \
              --config config.yaml"
        ;;

    late_switch)
        print_info "Running late switch experiment (epoch 150)"
        EXP_NAME="late_switch_150"
        ARGS="--experiment_name late_switch_150 \
              --switch_epoch 150 \
              --total_epochs 200 \
              --config config.yaml"
        ;;

    mean_mode)
        print_info "Running mean mode experiment (faster collapse expected)"
        EXP_NAME="mean_mode_collapse"
        ARGS="--experiment_name mean_mode_collapse \
              --synthetic_mode mean \
              --switch_epoch 100 \
              --total_epochs 200 \
              --config config.yaml"
        ;;

    sample_mode)
        print_info "Running sample mode experiment (default)"
        EXP_NAME="sample_mode_collapse"
        ARGS="--experiment_name sample_mode_collapse \
              --synthetic_mode sample \
              --switch_epoch 100 \
              --total_epochs 200 \
              --config config.yaml"
        ;;

    mixed_data)
        print_info "Running mixed data experiment (50% real + 50% synthetic)"
        EXP_NAME="mixed_data_50"
        ARGS="--experiment_name mixed_data_50 \
              --synthetic_ratio 0.5 \
              --switch_epoch 100 \
              --total_epochs 200 \
              --config config.yaml"
        ;;

    large_model)
        print_info "Running large model experiment"
        EXP_NAME="large_model_collapse"
        ARGS="--experiment_name large_model_collapse \
              --emsize 1024 \
              --nlayers 12 \
              --nhid 2048 \
              --nhead 8 \
              --switch_epoch 100 \
              --total_epochs 200"
        ;;

    custom)
        print_info "Running custom experiment"
        EXP_NAME="custom_experiment"
        ARGS="--experiment_name custom_experiment"

        # Add custom parameters
        [ ! -z "$SWITCH_EPOCH" ] && ARGS="$ARGS --switch_epoch $SWITCH_EPOCH"
        [ ! -z "$TOTAL_EPOCHS" ] && ARGS="$ARGS --total_epochs $TOTAL_EPOCHS"
        [ ! -z "$BATCH_SIZE" ] && ARGS="$ARGS --batch_size $BATCH_SIZE"
        ;;

    *)
        print_error "Unknown experiment type: $EXPERIMENT_TYPE"
        usage
        exit 1
        ;;
esac

# Check Python installation
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.7+"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "train_with_switch.py" ]; then
    print_error "train_with_switch.py not found. Please run from experiments/iterative_collapse/"
    exit 1
fi

# Print configuration
echo ""
print_info "Experiment Configuration:"
echo "  Type: $EXPERIMENT_TYPE"
echo "  Name: $EXP_NAME"
echo "  Device: $DEVICE"
echo ""

# Confirm before long runs
if [[ "$EXPERIMENT_TYPE" != "quick_test" ]]; then
    read -p "This will take several hours. Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Experiment cancelled"
        exit 0
    fi
fi

# Create directories
mkdir -p checkpoints metrics plots

# Run training
print_info "Starting training..."
echo ""

START_TIME=$(date +%s)

python train_with_switch.py $ARGS

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

print_success "Training complete!"
echo "  Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

# Generate visualizations
print_info "Generating visualizations..."

METRICS_FILE="metrics/${EXP_NAME}_metrics.json"

if [ -f "$METRICS_FILE" ]; then
    python visualize.py \
        --metrics_file "$METRICS_FILE" \
        --output_dir "plots"

    print_success "Visualizations saved to plots/"
    echo ""

    # Print summary
    print_info "Experiment Summary:"
    echo "  Metrics: $METRICS_FILE"
    echo "  Checkpoints: checkpoints/${EXP_NAME}_epoch_*.pt"
    echo "  Plots: plots/${EXP_NAME}_*.png"
    echo ""

    print_success "All done!"
else
    print_warning "Metrics file not found: $METRICS_FILE"
    print_warning "Visualization skipped"
fi
