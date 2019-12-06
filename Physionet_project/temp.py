import wfdb
import scipy.io
import os


path_to_data = 'H:\data\you-snooze-you-win-the-physionet-computing-in-cardiology-challenge-2018-1.0.0\\training\\tr03-0029'

item = 'tr03-0029'

combined = os.path.join(path_to_data, item)

# Arousal files (.mat)
a = scipy.io.loadmat(combined)

# Signal (.mat) and header (.hea) files
record = wfdb.rdrecord(combined)

# Arousal annotation files (.arousal)
annotation = wfdb.rdann(combined, 'arousal')

print(a)

print(record)

print(annotation)

print(record.__dict__)

# wfdb.plot_wfdb(record=record, annotation=annotation, plot_sym=True,
#                    time_units='seconds', title='MIT-BIH Record 100',
#                    figsize=(10,4))