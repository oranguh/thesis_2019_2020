import mne
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm

data_folder = Path("G:\\HMC22\\test")
signal_folder = data_folder / "edf"
annotation_folder = data_folder / "scorings"


def numpy_from_edf(path, signal_edf):

    total_length = signal_edf._data.shape[1]
    sfreq = edf_signal.info["sfreq"]
    seconds_to_hour = 60*60

    with open(path, "r") as f:
        for row in f:
            #       prints header
            print(row[:500])

            #       removes header
            #       splitting the events
            texts = row[500:].strip().rsplit("\x14")

            #       cleaning the data
            texts = [x.replace("\x00", "") for x in texts]
            texts = [x.replace("+", "") for x in texts]

            #       Splits onset and duration
            texts = [x.replace("\x15", ":") for x in texts]
            texts = [x.split(":") for x in texts]

    print("This session lasted {} hours \n".format(total_length / sfreq / seconds_to_hour))

    hypnogram = np.zeros(total_length)
    limb_movement = np.zeros(total_length)
    arousals = np.zeros(total_length)
    desaturation = np.zeros(total_length)
    central_apnea = np.zeros(total_length)
    hypopnea = np.zeros(total_length)

    cntr = Counter()

    for i, x in enumerate(texts):
        if i % 2 == 0:
            continue
        else:
            onset_duration = texts[i - 1]
            event_name = texts[i][0]

            if "Lights on" in event_name:
                continue
            elif "Lights off" in event_name:
                continue
            elif event_name == "":
                continue

            onset_duration = [float(x) for x in onset_duration]

            onset_index = int(onset_duration[0] * sfreq)
            duration = int(onset_duration[1] * sfreq)

            if event_name == "Sleep stage W":
                hypnogram[onset_index:onset_index + duration] = 1
            elif event_name == "Sleep stage R":
                hypnogram[onset_index:onset_index + duration] = 0
            elif event_name == "Sleep stage N1":
                hypnogram[onset_index:onset_index + duration] = -1
            elif event_name == "Sleep stage N2":
                hypnogram[onset_index:onset_index + duration] = -2
            elif event_name == "Sleep stage N3":
                hypnogram[onset_index:onset_index + duration] = -3

            elif "EEG arousal" in event_name:
                arousals[onset_index:onset_index + duration] = 1
            elif "Limb movement" in event_name:
                limb_movement[onset_index:onset_index + duration] = 1
            elif "Desaturation" in event_name:
                desaturation[onset_index:onset_index + duration] = 1
            elif "Central apnea" in event_name:
                central_apnea[onset_index:onset_index + duration] = 1
            elif "Hypopnea" in event_name:
                hypopnea[onset_index:onset_index + duration] = 1

            cntr[event_name] += 1

    annotation_dict = {"hypnogram": hypnogram,
                       "limb_Movement": limb_movement,
                       "arousals": arousals,
                       "desaturation": desaturation,
                       "central_Apnea": central_apnea,
                       "hypopnea": hypopnea}

    print(cntr)
    return annotation_dict

files = [x.parts[-1] for x in annotation_folder.glob("*.edf")]
print(files)

for file in tqdm(files):
    signal_path = signal_folder / file
    annotation_path = annotation_folder / file

    edf_signal = mne.io.read_raw_edf(signal_path, preload=True)
    annotation_dict = numpy_from_edf(annotation_path, edf_signal)