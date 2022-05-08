# README.md

## Quickstart

1. Make sure you are in the main folder of the repo where the .whl is, not in the submission
2. ```conda create --name radar_env python=3.8```
3. ```conda activate radar_env```
4. ```pip install ifxdaq-3.0.1-py3-none-xxx.whl``` Replace `xxx` with what is appropriate
5. ```pip install requirements.txt```

Hopefully it works. If not we hope it's not hard to solve

6. Relocate to ```submisson/code```
7. Run ```pipeline.ipynb``` to see results

The way we like to run the notebooks is by opening Anaconda navigator, changing the interpreter to `radar_env` and then
running the notebook.

If theres no jupyter ```pip install notebook```