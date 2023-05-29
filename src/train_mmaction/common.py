def load_label_map(file_path):
    """Load Label Map.
    Args:
        file_path (str): The file path of label map.
    Returns:
        dict: The label map (int -> label name).
    """
    # lines = open(file_path).readlines()
    # # lines = [x.strip().split(': ') for x in lines]
    # return {i+1: x for i, x in enumerate(lines)}
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


# Load label_map
def get_label(config):
    label_map = load_label_map(config["label_map"])
    try:
        if config['data']['train']['custom_classes'] is not None:
            label_map = {
                # id + 1: label_map[cls]
                cls: label_map[cls]
                for id, cls in enumerate(config['data']['train']
                                        ['custom_classes'])
            }
    except KeyError:
        pass
    return label_map