import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared_runner import run_attribute_analysis

if __name__ == "__main__":
	run_attribute_analysis('violence', sample_size='all')
