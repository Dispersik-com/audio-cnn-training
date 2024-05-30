import os
import argparse
from audio_utils import split_audio, remove_silent_segment
from data_utils import create_dataset
import random


def print_info(audio_data_dict, mfcc_size_dict, min_length_data):
    # output info
    def format_info(class_name):
        segment_list, samples_removed = audio_data_dict.get(class_name)
        mfcc_size = mfcc_size_dict.get(class_name)
        pth_file = f"{class_name}.pth"
        total_samples = len(segment_list)
        if min_length_data != 0:
            samples_removed_by_abs = total_samples - min_length_data
            samples_kept = total_samples - samples_removed_by_abs
        else:
            samples_kept = total_samples
            samples_removed_by_abs = 0
        return (class_name, total_samples + samples_removed, f"{total_samples} (-{samples_removed})",
                pth_file, f" {samples_kept} (-{samples_removed_by_abs})", str(list(map(lambda x: x, mfcc_size))))

    # get info
    data = [format_info(temp_dir) for temp_dir in audio_data_dict.keys()]

    # max_lengths = [max(len(str(item[i])) for item in data) for i in range(len(data[0]))]
    column_headers = ["Classes name", "Total of Samples", "After removed by db (removed)", "Torch files",
                      "Final data size (asb remove)", "Mfcc size"]

    max_lengths = [
        max(len(header), max(len(str(item[i])) for item in data))
        for i, header in enumerate(column_headers)
    ]

    # create table
    print("Dataset structure:")
    print("{:<{}} | {:<{}} | {:<{}} | {:<{}} | {:<{}} | {:<{}}".format(column_headers[0], max_lengths[0],
                                                              column_headers[1], max_lengths[1],
                                                              column_headers[2], max_lengths[2],
                                                              column_headers[3], max_lengths[3],
                                                              column_headers[4], max_lengths[4],
                                                              column_headers[5], max_lengths[5]))
    for item in data:
        print("{:<{}} | {:<{}} | {:<{}} | {:<{}} | {:<{}} | {:<{}}".format(item[0], max_lengths[0],
                                                                  item[1], max_lengths[1],
                                                                  item[2], max_lengths[2],
                                                                  item[3], max_lengths[3],
                                                                  item[4], max_lengths[4],
                                                                  item[5], max_lengths[5]))


def create_dataset_from_audiodata(dir_name, output_dir, save_segments, db_level_for_remove,
                                  segment_duration, n_mfcc, asb):
    audio_data_dict = {}

    if os.path.exists(output_dir):
        output_dir += " v1_" + str(random.randint(1, 100))

    for file_name in os.listdir(dir_name):
        if save_segments:
            save_output_dir = f"{output_dir}/{file_name}".split('.')[0]
        else:
            save_output_dir = None

        segment_list = split_audio(f"{dir_name}/{file_name}",
                                   save_output_dir=save_output_dir,
                                   segment_duration=segment_duration)

        segment_list, samples_removed = remove_silent_segment(segment_list,
                                                              threshold_db=db_level_for_remove)

        audio_data_dict[file_name.split(".")[0]] = (segment_list, samples_removed)

    _, mfcc_size_dict, min_length_data = create_dataset(audio_data_dict, output_dir, resize=asb, n_mfcc=n_mfcc)

    print_info(audio_data_dict, mfcc_size_dict, min_length_data)


def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dir', default="data", help='Input dir')
    parser.add_argument('--out', default=None, help='Output dir')
    parser.add_argument("--seg", action="store_true", help="Save segments")
    parser.add_argument('--dlf', default="-40", help='Maximum level for deletion fragments (in db)')
    parser.add_argument('--sd', default="0.5", help='Segment duration (in seconds)')
    parser.add_argument('--mfcc', default="20", help='Mel-frequency cepstral coefficients')
    parser.add_argument('--asb', action='store_true', help='Adjust data size for balance'
                                                           '(Warning: Deletes all data by smallest size)')

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    create_dataset_from_audiodata(args.dir, args.out, args.seg,
                                  db_level_for_remove=int(args.dlf),
                                  segment_duration=float(args.sd),
                                  n_mfcc=int(args.mfcc),
                                  asb=args.asb)


if __name__ == "__main__":
    main()
