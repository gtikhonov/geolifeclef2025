import os
import json

# Path to the config.json file
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

# Load configuration from config.json if it exists
config_data = {}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        config_data = json.load(f)

def get_config(key, default=None):
    """Retrieve configuration from environment variable or config.json."""
    return os.getenv(key, config_data.get(key, default))

# Base paths
DATA_PATH = get_config('GLC_DATA_PATH')
SCRATCH_PATH = get_config('GLC_SCRATCH_PATH')
LOCAL_SCRATCH = get_config('LOCAL_SCRATCH')

# HMSC specific
HMSC_HPC_PATH = get_config('HMSC_HPC_PATH')
PRITHVI_WEIGHTS_PATH = get_config('PRITHVI_WEIGHTS_PATH')

# Generic paths for root scripts
SOLUTIONS_DIR = get_config('SOLUTIONS_DIR')
OUTPUT_DIR = get_config('OUTPUT_DIR')

# Remote paths (for rsync)
MAHTI_SCRATCH = get_config('MAHTI_SCRATCH')
MAHTI_DATA_PATH = get_config('MAHTI_DATA_PATH')

# Validation: Warn if key paths are missing
if not DATA_PATH:
    import warnings
    warnings.warn("GLC_DATA_PATH is not set in environment or config.json")
