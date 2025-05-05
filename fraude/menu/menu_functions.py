def get_indexes(options):
    """
    Get the indexes for the menu options.
    """
    initial_options = {i: option for i, option in enumerate(options.keys(), start=1)}
    exit_option = {max(initial_options.keys(), default=0) + 1: 'Exit'}
    return initial_options | exit_option

def get_functions(options):
    """
    Get the functions for the menu options.
    """
    return {i: function for i, function in enumerate(options.values(), start=1)}

def display_menu(options):
    """
    Display the menu options.
    """
    print("Menu Options:")
    indexes = get_indexes(options)
    for i, option in indexes.items():
        print(f"{i}. {option}")

def get_choice(prompt, options, preselected_choice = None) -> int:
    """
    Get a valid choice from the user.
    """
    indexes = get_indexes(options).keys()
    choice = input(prompt) if preselected_choice is None else preselected_choice
    while not choice.isdigit() or int(choice) not in indexes:
        if not choice.isdigit():
            print("You must enter a number.")
        elif int(choice) not in indexes:
            print("Invalid choice. Please try again.")
        choice = input(prompt)
    return int(choice)

def menu(options, pre_start_choice = None, pre_end_choice = None) -> tuple[int, int] | tuple[None, None]:
    """
    Display the menu and get the user's choice.
    """
    exit_option = max(get_indexes(options).keys(), default=1)
    display_menu(options)
    start_choice = get_choice("Start from: ", options, preselected_choice=pre_start_choice)
    if start_choice == exit_option:
        print("Exiting...")
        return None, None
    end_choice = get_choice("End at: ", options, preselected_choice=pre_end_choice)
    while end_choice < start_choice:
        print("End choice must be greater than or equal to start choice.")
        end_choice = get_choice("End at: ", options)
    if end_choice == exit_option:
        print("Exiting...")
        return None, None
    return start_choice, end_choice