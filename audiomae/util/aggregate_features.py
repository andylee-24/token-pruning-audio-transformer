import os
import torch
from tqdm import tqdm
    


def combine_feature_dict(feature_dir, feature_name: str, load_subset: bool = False):
    
    # grep all files in feature_dir and select files start with feature_name
    

    file_paths = [
        os.path.join(root, filename)
        for root, dirs, files in os.walk(feature_dir)
        for filename in sorted(files)
        if filename.startswith(feature_name) and filename.endswith(".pth")
    ]
    
    if file_paths == []:
        return None
    
    if load_subset == True:
        file_paths = file_paths[:4]

    # Use multiprocessing Pool
    dict_list = []
    
    for filename in file_paths:
        d = torch.load(filename, weights_only=False)
        dict_list.append(d)
            
    if type(dict_list[0]) == torch.Tensor:
        output = torch.cat(dict_list, dim=0)
    elif type(dict_list[0]) == list:
        output = [d for dl in dict_list for d in dl]
    else:
        import pdb; pdb.set_trace()
        assert False

    return output