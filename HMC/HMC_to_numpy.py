import mne
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import pandas as pd

"""
HMC dataset has a lot of channels 27 in total. 
['EEG A2-Cz', 'EEG C4-Cz', 'EEG A1-Cz', 'EEG Fpz-Cz', 'EMG chin+', 'EOG E2-Cz', 'EEG F4-Cz', 'EEG O2-Cz', 'EEG F3-Cz', 
'EEG O1-Cz', 'EOG E1-Cz', 'EMG chin-', 'ECG', 'Resp oro-nasal', 'Sound', 'CPAP pressure', 'Resp nasal', 'Resp chest', 
'EMG RAT', 'Sound snore', 'Resp abdomen', 'EMG LAT', 'Body position', 'Accu', 'Pleth', 'Pulse', 'SaO2'] 
 
first 12 are on the head, we will use those.

There are also a lot of annotations
"hypnogram", "limb_Movement", "arousals", "desaturation", "central_Apnea", "hypopnea"}

We will only use the hypnogram and the arousals. 

Sampling frequency is 256 Hz, this will need to go down to 100
"""


def main():
    data_folder = Path("E:\\HMC22\\test")
    signal_folder = data_folder / "edf"
    annotation_folder = data_folder / "scorings"

    numpy_folder = data_folder / "numpy"
    numpy_folder.mkdir(parents=True, exist_ok=True)
    files = [x.parts[-1] for x in annotation_folder.glob("*.edf")]

    for file in tqdm(files, total=len(files)):
        signal_path = signal_folder / file
        annotation_path = annotation_folder / file
        numpy_folder_to_save = numpy_folder / file
        numpy_folder_to_save = numpy_folder_to_save.with_suffix("")
        numpy_folder_to_save.mkdir(parents=True, exist_ok=True)

        edf_signal = mne.io.read_raw_edf(signal_path, preload=True)
        annotation_dict = numpy_from_edf(annotation_path, edf_signal)
        print(file)
        print(annotation_dict["arousals"].shape, annotation_dict["hypnogram"].shape, edf_signal._data.shape)

        arousals = resample_signal(annotation_dict["arousals"], 256, 100)
        hypnogram = resample_signal(annotation_dict["hypnogram"], 256, 100)

        length_signal = len(arousals)
        labels = np.zeros((2, length_signal))
        data = np.zeros((12, length_signal))

        labels[0] = arousals
        labels[1] = hypnogram

        for i in range(12):
            data[i] = resample_signal(edf_signal._data[i], 256, 100)

        np.save(numpy_folder_to_save / file.replace(".edf", "_data"), data)
        np.save(numpy_folder_to_save / file.replace(".edf", "_labels"), labels)



def resample_signal(signal, input_freq, output_freq):
    """
    resamples data
    """
    ms = str(round(1 / input_freq, 6))

    input_date_range = pd.date_range(0, periods=len(signal), freq=ms + 'S')

    df = pd.DataFrame(signal, index=input_date_range)

    ms = str(round(1 / output_freq, 6))

    output_size = (len(signal) / input_freq) * output_freq
    output_date_range = pd.date_range(0, periods=output_size, freq=ms + 'S')

    df = df.reindex(output_date_range, method='nearest')

    output_signal = df[0].to_numpy()
    return output_signal


def numpy_from_edf(path, signal_edf):

    total_length = signal_edf._data.shape[1]
    sfreq = signal_edf.info["sfreq"]
    seconds_to_hour = 60*60

    with open(path, "r") as f:
        for row in f:
            #       prints header
            # print(row[:500])

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

    # print(cntr)
    return annotation_dict

if __name__ == '__main__':
    main()
