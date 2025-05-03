from fraude import get_fraude_dataset
from pathlib import Path
import pandas as pd

def main():
    project_path = Path('.').resolve()
    fraude = get_fraude_dataset(project_path)

if __name__ == "__main__":
    main()