"""
Arousal
Hypopnea
SpO2_desaturation
SpO2_artifact
Hypnogram (wake, REM, 1,2,3)
Obstructive apnea
central_apnea

The sampling rate for all of these will be the same. Namely 10 Hz. Since you can have tenth of a second
The full stream of data will be determined by the duration of the recording start time in seconds * 10. e.g. 32520 * 10
"""
import xml.etree.ElementTree as ET
import numpy as np
import os
import pyedflib
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
import random
import platform


def main():

    if platform.system() == 'Windows':
        patient_records_path = "F:\\shhs\\datasets\\shhs1-dataset-0.15.0.csv"

        arousal_xml_directory = 'F:\\shhs\\polysomnography\\annotations-events-nsrr\\shhs1'
        edf_directory = 'F:\\shhs\\polysomnography\\edfs\\shhs1'
        save_folder = "F:\\shhs\\polysomnography\\shh1_numpy\\"
    else:
        patient_records_path = "/project/marcoh/shhs/datasets/shhs1-dataset-0.15.0.csv"

        arousal_xml_directory = '/project/marcoh/shhs/polysomnography/annotations-events-nsrr/shhs1'
        edf_directory = '/project/marcoh/shhs/polysomnography/edfs/shhs1'
        save_folder = "/project/marcoh/shhs/polysomnography/shh1_numpy"

    Path(save_folder).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(patient_records_path)

    partition_list = np.asarray([])

    for subject_id in tqdm(df.nsrrid, total=len(df.nsrrid)):
        subject_id = str(subject_id)
        arousal_xml_path_ = "shhs1-" + subject_id + "-nsrr.xml"
        arousal_xml_path = os.path.join(arousal_xml_directory, arousal_xml_path_)

        edf_path_ = "shhs1-" + subject_id + ".edf"
        edf_path = os.path.join(edf_directory, edf_path_)

        if os.path.exists(arousal_xml_path):
            save_folder_individual = os.path.join(save_folder, subject_id)
            Path(save_folder_individual).mkdir(parents=True, exist_ok=True)

            if os.path.exists(os.path.join(save_folder_individual, subject_id) + "_labels.npy"):
                partition_list = np.append(partition_list, subject_id)
                print("Data already saved, skipping ", subject_id)

            else:
                info_list, signal_list = subject_to_edf(arousal_xml_path, edf_path)
                np.save(os.path.join(save_folder_individual, subject_id) + "_data", signal_list[:-2, :])
                np.save(os.path.join(save_folder_individual, subject_id) + "_labels", signal_list[-2:, :])
                partition_list = np.append(partition_list, subject_id)
                print("Saved")
        else:
            print("{} does not exist, skipping".format(arousal_xml_path))

    print('\n', partition_list, '\n\n')
    print("partition made with {} elements".format(len(partition_list)))
    random.shuffle(partition_list)
    split_point = int(np.round(len(partition_list) / 3))

    partition = {'validation': [str(x) for x in partition_list[:split_point]],
                 'train': [str(x) for x in partition_list[split_point:]]}

    save_obj(partition, os.path.join(save_folder, 'data_partition'))

    print('Training: ', partition['train'], '\nValidation: ', partition['validation'])


def create_signal_from_SHHS_xml(arousal_xml_path):
    sample_rate = 10

    tree = ET.parse(arousal_xml_path)
    root = tree.getroot()
    events = list(root)[-1]

    recording_duration = int(float(events[0][3].text) * sample_rate)

    arousal, hypopnea, spO2_desaturation, spO2_artifact, hypnogram, obstructive_apnea = np.zeros(
        recording_duration), np.zeros(recording_duration), np.zeros(recording_duration), np.zeros(
        recording_duration), np.zeros(recording_duration), np.zeros(recording_duration)
    central_apnea = np.zeros(recording_duration)

    for child in events:

        event = str(child[1].text).split("|")[0]
        event_start = int(float(child[2].text) * sample_rate)
        event_duration = int(float(child[3].text) * sample_rate)
        event_end = event_start + event_duration

        # print(event)
        if event == "Recording Start Time":
            continue

        elif event == "Arousal":
            arousal[event_start:event_end] = 1

        elif event == "Hypopnea":
            hypopnea[event_start:event_end] = 1

        elif event == "SpO2 desaturation":
            spO2_desaturation[event_start:event_end] = 1

        elif event == "SpO2 artifact":
            spO2_artifact[event_start:event_end] = 1

        elif event == "Obstructive apnea":
            obstructive_apnea[event_start:event_end] = 1

        elif event == "Central apnea":
            central_apnea[event_start:event_end] = 1

        elif event == "Wake":
            hypnogram[event_start:event_end] = 1

        elif event == "REM sleep":
            hypnogram[event_start:event_end] = 0

        elif event == "Stage 1 sleep":
            hypnogram[event_start:event_end] = -1

        elif event == "Stage 2 sleep":
            hypnogram[event_start:event_end] = -2

        elif event == "Stage 3 sleep":
            hypnogram[event_start:event_end] = -3

        elif event == "Stage 4 sleep":
            hypnogram[event_start:event_end] = -4

        elif event == "Mixed apnea":
            pass
        elif event == "Unsure":
            pass
        elif event == "Arousal resulting from Chin EMG":
            pass
        elif event == "ASDA arousal":
            pass
        elif event == "Unscored":
            pass
        else:
            print("{} event type not found!".format(event))

    #     annotation_names = ['arousal', 'hypopnea', 'spO2_desaturation', 'spO2_artifact', 'hypnogram', 'obstructive_apnea', 'central_apnea']
    #     annotation_list = [arousal, hypopnea, spO2_desaturation, spO2_artifact, hypnogram, obstructive_apnea, central_apnea]

    annotation_names = ['arousal', 'hypnogram']
    annotation_list = [arousal, hypnogram]

    return annotation_list, annotation_names


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


def subject_to_edf(arousal_xml_path, edf_path):
    """
    returns the signal and annotations as numpy, also resamples

    I made it so I only use the channels ["EEG", "EEG(sec)", "EMG"]
    OMG this dataset is... ugh. the second electrode for EEG can be called "EEG 2" or "EEG(sec)" or "EEG(SEC)" or "EEG2"
    who why when who
    EEG (sec): 	C3 	A2
    EEG:    C4 	A1
    Annotations only contain ['arousal', 'hypnogram']

    """
    print("\nCreating subject {}".format(arousal_xml_path.strip("-nsrr.xml")))
    TARGET_FREQ = 100
    channel_info = []
    data_list = []

    MAX_DATA = 0

    with pyedflib.EdfReader(edf_path) as f:

        for channel in list(range(f.signals_in_file)):
            print(f.getLabel(channel))
            # Are these always in the same order? Order seems to be: EEG(sec), EMG, EEG
            # Why are there so many variants for the second EEG
            if f.getLabel(channel) in ["EEG", "EMG", "EEG 2", "EEG2", "EEG(SEC)", "EEG(sec)", "EEG sec"]:

                ch_dict = {'label': f.getLabel(channel), 'dimension': f.getPhysicalDimension(channel),
                           'sample_rate': f.getSampleFrequency(channel), 'physical_max': f.getPhysicalMaximum(channel),
                           'physical_min': f.getPhysicalMinimum(channel), 'digital_max': f.getDigitalMaximum(channel),
                           'digital_min': f.getDigitalMinimum(channel), 'transducer': f.getTransducer(channel),
                           'prefilter': f.getPrefilter(channel)}

                channel_info.append(ch_dict)
                data_list.append(resample_signal(f.readSignal(channel), f.getSampleFrequency(channel), TARGET_FREQ))

            if MAX_DATA < f.getNSamples()[channel]:
                MAX_DATA = f.getNSamples()[channel]

        if len(data_list) != 3:
            print("{} does not have 3 channels".format(arousal_xml_path.strip("-nsrr.xml")))
            print(channel_info)
    # Sample rate for the annotations are always 10

    sample_rate = 10

    annotation_list, annotation_names = create_signal_from_SHHS_xml(arousal_xml_path)

    # Creating the annotation signals
    for i, annotation in enumerate(annotation_list):
        ch_dict = {'label': annotation_names[i], 'dimension': "",
                   'sample_rate': sample_rate, 'physical_max': 100, 'physical_min': -100,
                   'digital_max': 32768, 'digital_min': -32768, 'transducer': '', 'prefilter': ''}

        channel_info.append(ch_dict)

        data_list.append(resample_signal(annotation, sample_rate, TARGET_FREQ))

    """
    creating combined EDF format for visualization in polyman
    """
    polyman = False
    if polyman:
        name_ = arousal_xml_path.strip('-nsrr.xml')
        name_ += '-combined.edf'

        test_data_file = os.path.join('.', name_)

        with pyedflib.EdfWriter(test_data_file, len(channel_info), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
            # print(channel_info)
            # print(data_list)
            f.setSignalHeaders(channel_info)
            f.writeSamples(data_list)

    print("\nFinished subject {}".format(arousal_xml_path.strip("-nsrr.xml")))

    return channel_info, np.array(data_list)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)
# arousal_xml_path = 'F:\\shhs\\polysomnography\\annotations-events-nsrr\\shhs1\\shhs1-200001-nsrr.xml'
# edf_path = 'F:\\shhs\\polysomnography\\edfs\shhs1\\shhs1-200001.edf'
#
# info_list, signal_list = subject_to_edf(arousal_xml_path, edf_path)
# print(signal_list.shape)


if __name__ == '__main__':
    main()
