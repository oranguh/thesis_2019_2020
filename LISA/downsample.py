import csv
import torch
import os

folder_path = '/project/marcoh/you_snooze_you_win/marco/'
records = '/project/marcoh/you_snooze_you_win/marco/RECORDS'
# stopped at "tr10-0392"
skip = True
with open(records, "r") as f:
    for line in csv.reader(f):
        individual = line[0].strip("/")
        if individual == "tr10-0413":
            print("found breakpoint", individual, "tr10-0413")
            skip = False

        if skip:
            print("skipping ", individual)
            continue
        else:
            individual_folder = os.path.join(folder_path, individual)
            individual_records_ = os.path.join(individual_folder, individual)

            X = torch.load(individual_records_ + '_data.pt')
            Y = torch.load(individual_records_ + '_labels.pt')
            X = X[:, ::2]
            Y = Y[:, ::2]
            torch.save(X, individual_records_ + '_data.pt')
            torch.save(Y, individual_records_ + '_labels.pt')
            print("finished ", individual)