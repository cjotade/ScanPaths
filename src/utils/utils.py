import numpy as np

def cast_to_int(x):
    try:
        return int(x)
    except:
        return None

def create_empty_dict(arr_keys):
    dict_keys = dict.fromkeys(arr_keys)
    for key in dict_keys.keys():
        dict_keys[key] = []
    return dict_keys

def merge_dicts(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def round_parser(val_arr):
    round_up_idx = np.where(val_arr % 1 >= 0.5)[0]
    round_down_idx = np.where(val_arr % 1 < 0.5)[0]
    val_arr[round_up_idx] = np.ceil(val_arr[round_up_idx])
    val_arr[round_down_idx] = np.round(val_arr[round_down_idx])
    return val_arr.astype(int)

def split_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]