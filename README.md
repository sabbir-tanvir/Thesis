# Thesis Project

## Overview
This project involves...

## New Functionality: Reading Sample_Small Text Files

A new function `read_sample_small` has been added to read and process text files from the `Sample_Small` directory.

### Usage
To read the text files from the `Sample_Small` directory, use the following code:

```python
sample_small_dir = '/c:/Users/Sabbir/Documents/GitHub/Thesis/Sample_Small'
sample_data = read_sample_small(sample_small_dir)
print(f"Loaded {len(sample_data)} text files from Sample_Small directory.")
```

This will load the content of all `.txt` files in the `Sample_Small` directory and its subdirectories into a list called `sample_data`.
```
