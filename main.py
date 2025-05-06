import click
from pathlib import Path

@click.command()
@click.option('--project-path', default=Path('.').resolve(), help='Path to the project directory.')
@click.option('--start-from', default=None, help='Start choice for the menu.')
@click.option('--end-at', default=None, help='End choice for the menu.')
def train(project_path, start_from, end_at):
    _train(project_path=project_path, start_from=start_from, end_at=end_at)

def _train(project_path, start_from, end_at):
    from fraude.pipelines import get_data_pipeline, clean_data_pipeline, add_features_pipeline, train_model_pipeline, evaluate_model_pipeline, predict_pipeline, single_predict_pipeline
    from fraude.menu import menu, get_functions
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
    prediction_path = _inference(project_path)
    print(f"Predictions saved to {prediction_path}")

def _inference(project_path: Path):
    from fraude.pipelines import get_data_pipeline, clean_data_pipeline, add_features_pipeline, train_model_pipeline, evaluate_model_pipeline, predict_pipeline, single_predict_pipeline
    from fraude.menu import menu, get_functions
    _, prediction_path = predict_pipeline(project_path)
    return prediction_path

@click.command()
@click.option('--project-path', default=Path('.').resolve(), help='Path to the project directory.')
@click.option('--amount', default=1.0, help='Amount of predictions to make.')
@click.option('--type', default='single', help='Type of prediction: single or batch.')
def single_inference(project_path, amount, type):
    """
    Run a single inference pipeline.
    """
    project_path = Path(project_path).resolve()
    y = _single_inference(project_path, amount, type)
    prediction = y[0] if len(y) > 0 else None
    prediction_text = "Fraude" if prediction == 1 else "No fraude" if prediction == 0 else "No prediction made"
    print(f"Single prediction result: {prediction_text}")

def _single_inference(project_path, amount, type):
    """
    Run a single inference pipeline.
    """
    from fraude.pipelines import get_data_pipeline, clean_data_pipeline, add_features_pipeline, train_model_pipeline, evaluate_model_pipeline, predict_pipeline, single_predict_pipeline
    from fraude.menu import menu, get_functions
    project_path = Path(project_path).resolve()
    y = single_predict_pipeline(project_path, amount, type)
    return y

if __name__ == "__main__":
    # _train(project_path=Path('.').resolve(),start_from="5", end_at="5")
    #_single_inference(project_path=Path('.').resolve(), amount=24074, type='TRANSFER')
    _inference(project_path=Path('.').resolve())