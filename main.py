import click
from fraude import get_fraude_dataset, clean_column_names, fix_datetime_columns, remove_outliers, remove_zeros
from fraude import save_cleaned_data, save_features_data
from fraude import add_week_day, split_city
from fraude.pipelines import get_data_pipeline, clean_data_pipeline, add_features_pipeline
from fraude.menu import menu, get_functions
from pathlib import Path

@click.command()
@click.option('--project-path', default=Path('.').resolve(), help='Path to the project directory.')
@click.option('--start-from', default=None, help='Start choice for the menu.')
@click.option('--end-at', default=None, help='End choice for the menu.')
def main(project_path, start_from, end_at):
    project_path = Path(project_path).resolve()
    options = {
        'Get Data': lambda: get_data_pipeline(project_path),
        'Clean Data': lambda: clean_data_pipeline(project_path),
        'Add Features': lambda: add_features_pipeline(project_path)
    }
    start_choice, end_choice = menu(options, pre_start_choice=start_from, pre_end_choice=end_at)
    if start_choice is None or end_choice is None:
        return
    functions = get_functions(options)
    print(f"Running from {start_choice} to {end_choice}...")
    for i in range(start_choice, end_choice + 1):
        functions[i]()

def menu_():
    options = {
        'Get Data': get_data_pipeline,
        'Clean Data': clean_data_pipeline,
        'Add Features': add_features_pipeline
    }
    indexes = {i: option for i, option in enumerate(options.keys(), start=1)}
    for i, option in indexes.items():
        print(f"{i}. {option}")
    print("4. Exit")
    
    start_choice = input("Run from: ")
    while start_choice not in ['1', '2', '3', '4', '5', '6']:
        print("Invalid choice. Please try again.")
        start_choice = input("Run from: ")
    end_choice = input("End at: ")
    while end_choice not in ['1', '2', '3', '4', '5', '6'] or end_choice < start_choice:
        if end_choice not in ['1', '2', '3', '4', '5', '6']:
            print("Invalid choice. Please try again.")
        elif end_choice < start_choice:
            print("End choice must be greater than or equal to start choice.")
        end_choice = input("End at: ")
    return start_choice, end_choice

if __name__ == "__main__":
    main()