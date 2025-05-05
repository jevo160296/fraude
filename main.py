import click
from fraude import get_fraude_dataset, clean_column_names, fix_datetime_columns, remove_outliers, remove_zeros
from fraude import save_cleaned_data, save_features_data
from fraude import add_week_day, split_city
from fraude.pipelines import get_data_pipeline, clean_data_pipeline, add_features_pipeline, train_model_pipeline, evaluate_model_pipeline, predict_pipeline
from fraude.menu import menu, get_functions
from pathlib import Path

@click.command()
@click.option('--project-path', default=Path('.').resolve(), help='Path to the project directory.')
@click.option('--start-from', default=None, help='Start choice for the menu.')
@click.option('--end-at', default=None, help='End choice for the menu.')
def train(project_path, start_from, end_at):
    _train(project_path=project_path, start_from=start_from, end_at=end_at)

def _train(project_path, start_from, end_at):
    project_path = Path(project_path).resolve()
    options = {
        'Get Data': lambda: get_data_pipeline(project_path),
        'Clean Data': lambda: clean_data_pipeline(project_path),
        'Add Features': lambda: add_features_pipeline(project_path),
        'Train Model': lambda: train_model_pipeline(project_path),
        'Evaluate Model': lambda: evaluate_model_pipeline(project_path)
    }
    start_choice, end_choice = menu(options, pre_start_choice=start_from, pre_end_choice=end_at)
    if start_choice is None or end_choice is None:
        return
    functions = get_functions(options)
    print(f"Running from {start_choice} to {end_choice}...")
    for i in range(start_choice, end_choice + 1):
        functions[i]()
    print("All pipelines executed successfully.")

@click.command()
@click.option('--project-path', default=Path('.').resolve(), help='Path to the project directory.')
def inference(project_path):
    """
    Run the inference pipeline.
    """
    project_path = Path(project_path).resolve() 
    _, prediction_path = predict_pipeline(project_path)
    print(f"Predictions saved to {prediction_path}")

if __name__ == "__main__":
    # _train(project_path=Path('.').resolve(),start_from="5", end_at="5")
    inference()