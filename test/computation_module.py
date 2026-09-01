from pathlib import Path


def add(x, y):
    return x + y


def multiply(x, y):
    return x * y


def subtract(x, y):
    return x - y


def affine(x, a=1, b=0):
    return a * x + b


def load_data(file_path):
    print(f"Loading data from {file_path}, abs: {Path(file_path).resolve().absolute()}")
    import pandas as pd

    try:
        data = pd.read_csv(Path(file_path).resolve().absolute())
        return data["value"]  # return numpy array
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"An error occurred while loading data: {e}") from e


def save_data(data, file_path):
    try:
        data.to_csv(file_path, index=True)
    except Exception as e:
        raise ValueError(f"An error occurred while saving data: {e}") from e
