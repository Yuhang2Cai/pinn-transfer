import pandas as pd
import os
from pathlib import Path


def convert_xlsx_to_csv(root_folder):
    """
    Convert all .xlsx files in the given folder and subfolders to .csv files
    """
    # Define the main directory paths
    data_root = Path(root_folder)
    subdirs = ['train', 'val', 'test']

    for subdir in subdirs:
        folder_path = data_root / subdir

        # Check if the directory exists
        if folder_path.exists():
            print(f"Processing {subdir} directory...")

            # Find all xlsx files in the directory
            for xlsx_file in folder_path.rglob("*.xlsx"):
                try:
                    # Read the Excel file
                    df = pd.read_excel(xlsx_file)

                    # Create CSV filename by replacing extension
                    csv_file = xlsx_file.with_suffix('.csv')

                    # Save as CSV
                    df.to_csv(csv_file, index=False)
                    print(f"Converted: {xlsx_file.name} -> {csv_file.name}")

                except Exception as e:
                    print(f"Error converting {xlsx_file}: {str(e)}")
        else:
            print(f"Directory {subdir} not found")


# Usage
root_directory = "D:/project/pinnTrasfer/data"
convert_xlsx_to_csv(root_directory)
print("Conversion completed!")
