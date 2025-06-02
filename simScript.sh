#! /bin/bash
source venv/bin/activate
python MCDataGenerator.py 3000
python3 DataGeneratorFromPythonMCs.py
