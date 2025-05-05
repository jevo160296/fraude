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