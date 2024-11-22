# Directories and files to clean
WANDB_DIR = wandb
CACHE_DIR = __pycache__
LOGS_DIR = logs
CHECKPOINTS_DIR = checkpoints
SLURM_OUTPUT_FILES = gpu_triplet_ssl_output.* gpu_vpr_ce_output.*
SLURM_ERROR_FILES = gpu_triplet_ssl_error.* gpu_vpr_ce_error.*

# Default target
.PHONY: all
all:
	@echo "Available commands:"
	@echo "  make clean        - Remove all generated files and directories"
	@echo "  make clean-wandb  - Remove only wandb files"
	@echo "  make clean-cache  - Remove Python cache files"
	@echo "  make clean-logs   - Remove log files"
	@echo "  make clean-ckpt   - Remove checkpoint files"
	@echo "  make clean-slurm  - Remove SLURM output and error files"

# Clean all generated files
.PHONY: clean
clean: clean-wandb clean-cache clean-logs clean-ckpt clean-slurm
	@echo "Cleaned all generated files"

# Clean wandb files
.PHONY: clean-wandb
clean-wandb:
	@echo "Cleaning wandb files..."
	@rm -rf $(WANDB_DIR)/*offline-run*
	@rm -rf $(WANDB_DIR)/debug*
	@rm -rf $(WANDB_DIR)/run*
	@echo "Wandb files cleaned"

# Clean Python cache
.PHONY: clean-cache
clean-cache:
	@echo "Cleaning Python cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*.pyd" -delete
	@echo "Python cache cleaned"

# Clean log files
.PHONY: clean-logs
clean-logs:
	@echo "Cleaning log files..."
	@rm -rf $(LOGS_DIR)/*.log
	@rm -rf $(LOGS_DIR)/*.txt
	@echo "Log files cleaned"

# Clean checkpoint files
.PHONY: clean-ckpt
clean-ckpt:
	@echo "Cleaning checkpoint files..."
	@rm -rf $(CHECKPOINTS_DIR)/*.pt
	@rm -rf $(CHECKPOINTS_DIR)/*.pth
	@echo "Checkpoint files cleaned"

# Clean SLURM output and error files
.PHONY: clean-slurm
clean-slurm:
	@echo "Cleaning SLURM output and error files..."
	@rm -f $(SLURM_OUTPUT_FILES)
	@rm -f $(SLURM_ERROR_FILES)
	@echo "SLURM files cleaned"

# Create necessary directories
.PHONY: init
init:
	@mkdir -p $(WANDB_DIR)
	@mkdir -p $(LOGS_DIR)
	@mkdir -p $(CHECKPOINTS_DIR)
	@echo "Directories created"