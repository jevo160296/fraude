class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    def stylize(text: str, style: str):
        return f"{style}{text}{bcolors.ENDC}"
    
class theme:
    def pipeline_status(pipeline_name: str, status: str):
        """
        Generates a stylized status message for a given pipeline.
        Args:
            pipeline_name (str): The name of the pipeline.
            status (str): The current status of the pipeline.
        Returns:
            str: A stylized string indicating the pipeline name and its status.
        """
        return bcolors.stylize(f"{pipeline_name} pipeline: {status}", bcolors.OKGREEN)

    def menu_option(option: str, description: str):
        """
        Generates a stylized menu option.
        Args:
            option (str): The menu option.
            description (str): The description of the menu option.
        Returns:
            str: A stylized string indicating the menu option and its description.
        """
        return bcolors.stylize(f"{option}: {description}", bcolors.OKCYAN)

    def header(text: str):
        """
        Generates a stylized header.
        Args:
            text (str): The header text.
        Returns:
            str: A stylized string indicating the header.
        """
        return bcolors.stylize(text, bcolors.HEADER)

def validate_dataset(df, expected_columns):
    """
    Validate the dataset against expected columns.
    
    Args:
        df (pd.DataFrame): The DataFrame to validate.
        expected_columns (list): List of expected column names.
        
    Returns:
        bool: True if the dataset is valid, False otherwise.
    """
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        print(bcolors.stylize("Missing columns:", bcolors.FAIL), {missing_columns})
        return False
    return True