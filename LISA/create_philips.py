from dry_dry_sham import ACDryDry
import random
import numpy as np
import pickle as pkl


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

# Not sure where to save the partition file, so whatever.
SAVEPATH = "/home/017320_arousal_data/"


p = ACDryDry()
partition_list = p.recording_ids
random.shuffle(partition_list)

split_point = int(np.round(len(partition_list) / 3))

partition = {'validation': [str(x) for x in partition_list[:split_point]],
             'train': [str(x) for x in partition_list[split_point:]]}

save_obj(partition, os.path.join(save_folder, 'data_partition.pkl'))

print('Training: ', partition['train'], '\nValidation: ', partition['validation'])
