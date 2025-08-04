# -------------------------- MONUMENT NAME EXTRACTION ------------------------ #

# ---------------------------------------------------------------------------- #

def clean_name(name: str) -> str:
    """
    Cleans and normalizes the text
    Args:
        name (str): the text to clean
    Returns:
        str: the cleaned text
    """
    return name.strip().lower()