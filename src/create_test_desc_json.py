"""
Use this script to create JSON-Line description files that can be used to
train deep-speech models through this library.
This works with data directories that are organized like LibriSpeech:
data_directory/group/speaker/[file_id1.wav, file_id2.wav, ...,
                              speaker.trans.txt]

Where speaker.trans.txt has in each line, file_id transcription

from https://github.com/baidu-research/ba-dls-deepspeech/blob/master/create_desc_json.py
"""


import argparse
import json
import os
import wave
import re


def main(data_directory, output_file):
    labels = []
    durations = []
    keys = []
    for dataset in os.listdir(data_directory):
        speaker_path = os.path.join(data_directory, dataset)
        labels_file = os.path.join(speaker_path,
                                   '{}.trans.txt'
                                   .format(dataset[:-9]))
        file_id = 1
        for line in open(labels_file):
            label = re.findall('\w+', line)
            label = ' '.join(label)
            audio_file = os.path.join(speaker_path,
                                    dataset[:-8] + str(file_id)) + '.wav'
            audio = wave.open(audio_file)
            duration = float(audio.getnframes()) / audio.getframerate()
            audio.close()
            keys.append(audio_file)
            durations.append(duration)
            labels.append(label)
            file_id += 1
    with open(output_file, 'w') as out_file:
        for i in range(len(keys)):
            line = json.dumps({'key': keys[i], 'duration': durations[i],
                              'text': labels[i]})
            out_file.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')
    parser.add_argument('output_file', type=str,
                        help='Path to output file')
    args = parser.parse_args()
    main(args.data_directory, args.output_file)
