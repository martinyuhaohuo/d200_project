

def read_txt(text_dir):
    """
    This function reads all texts in a folder and store them in a dictionary

    Parameters:
    -----------
    text_dir : str
        designating the path to the folder of txt files

    Returns:
    --------
    dict
        a dictionary with key as publishing date of the text, value as content of the text
    """
    texts = {}
    for path in text_dir.iterdir():
        with open(path, "r") as file:
            file_name = path.name[:-4]
            publish_time = file_name[:4] + "-" + file_name[4:6] + "-" + file_name[6:]
            texts[publish_time] = file.read()
    return texts