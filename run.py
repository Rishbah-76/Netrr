#!/usr/bin/env python
"""
Smart AI Glasses for the Visually Impaired
Standalone runner script
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Make sure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the main function from the package
from smart_glasses import main

if __name__ == "__main__":
    main() 