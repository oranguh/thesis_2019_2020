from pathlib import Path
import pickle as pkl
from tqdm import tqdm
import hashlib
import random


def main():

    folder_to_encrypt = Path("/home/017320_arousal_data/SHHS/")
    all_records = sorted(folder_to_encrypt.glob("*"))
    hash_records(all_records, folder_to_encrypt)


def hash_records(all_records, folder_to_encrypt):

    partition_list = []
    for i, record in tqdm(enumerate(all_records), total=len(all_records)):
        if record.is_dir():

            hash = hashlib.sha1(bytes(record.name, encoding='utf8')).hexdigest()
            for suffix in ["_data.npy", "_labels.npy"]:
                pass
                data_filename = record.name + suffix
                data = record / data_filename

                data_filename = hash + suffix
                data.rename(record / data_filename)

            record.rename(record.parent / hash)
            partition_list += hash

    random.shuffle(partition_list)
    split_point = int(len(all_records) / 3)

    partition = {'validation': [x for x in all_records[:split_point]],
                 'train': [x for x in all_records[split_point:]]}

    print(partition)
    save_obj(partition, folder_to_encrypt / "partition.pkl")


def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()





