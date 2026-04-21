import os
import shutil

from tqdm import tqdm

import numpy as np
import pandas as pd

import cv2

import argparse


# default location of cholec80 dataset
default_src_dir = r"/media/ExtHDD/cholec80"
default_output_dir = r"/media/ExtHDD/cholec80_data"
default_time_step = 25
default_test_size = 0.5
default_shuffle = False

default_label_dir = os.path.dirname(os.path.abspath(__file__))

# for build the labels
phase_label_files = os.path.join(default_src_dir, "phase_annotations")
tool_label_files = os.path.join(default_src_dir, "tool_annotations")

train_set_info_csv_loc = os.path.join(default_label_dir, "train_set_info.csv")
test_set_info_csv_loc = os.path.join(default_label_dir, "test_set_info.csv")


def get_phase_label(video_name, frame_index):
    phase_label_txt_info = pd.read_csv(os.path.join(phase_label_files, video_name+"-phase.txt"), sep="\t")
    
    return phase_label_txt_info.loc[phase_label_txt_info["Frame"]==frame_index, "Phase"].values[0]


def dataset_split(src_dir=None, output_dir=None, time_step=None, size=None, shuffle=None):
    # default paras
    src_dir = default_src_dir if src_dir is None else src_dir
    output_dir = default_output_dir if output_dir is None else output_dir
    time_step = default_time_step if time_step is None else default_time_step
    size = default_test_size if size is None else default_test_size
    shuffle = default_shuffle if shuffle is None else default_shuffle

    # create the train and test folders to store data
    for folder in ['train', 'test']:
        folder_path = os.path.join(output_dir, folder)
        if not os.path.exists(folder_path):
            pass
        else:
            print('Prepare to delete folder {}'.format(folder_path))
            print('Really? To continue, Enter y. To exit, Enter n!')
            if input() == "y":
                shutil.rmtree(folder_path)
                print('Folder {} is deleted'.format(folder_path))
                os.mkdir(folder_path)
                print('Folder {} is created'.format(folder_path))

    # initialization
    train_set_info = pd.DataFrame(None, columns=["video_name", "Frame", "file_loc", "label"])
    test_set_info = pd.DataFrame(None, columns=["video_name", "Frame", "file_loc", "label"])
    
    # reading the videos
    videos = [x for x in os.listdir(os.path.join(src_dir, "videos")) if x.__contains__(".mp4")]
    # sorted the videos
    videos.sort()
    
    # shuffling the videos
    if shuffle:
        np.random.shuffle(videos)
        
    # certifying the dividing point
    size = 0.5
    split_trainset_size = int(len(videos)*(1-size))

    # travesing all of the videos and extracting the frames
    for idx in tqdm(range(len(videos))):
        video_path = os.path.join(src_dir, "videos", videos[idx])
        video_fd = cv2.VideoCapture(video_path)

        if not video_fd.isOpened():
            print('Skipped: {}'.format(video_path))
            continue

        # the first 40 videos set as train set while the last 40 videos set as test set
        # there are 80 videos in cholec80 dataset
        video_type = 'train' if idx < split_trainset_size else 'test'

        # reading the frames
        frame_index, count_temp = 0, 0
        success, frame = video_fd.read()
        video_name = videos[idx].rsplit('.')[0]
        while success:
            if frame_index % time_step == 0:
                img_path = os.path.join(
                    output_dir, video_type, '%s_%d.jpg' % (video_name, frame_index))
                cv2.imwrite(img_path, frame)
                
                # saving the infos of frames
                if video_type == 'train':
                    train_set_info = train_set_info.append({"video_name":video_name, "Frame":frame_index, "file_loc":img_path, "label":get_phase_label(video_name, frame_index)}, ignore_index=True)
                else:
                    test_set_info = test_set_info.append({"video_name":video_name, "Frame":frame_index, "file_loc":img_path, "label":get_phase_label(video_name, frame_index)}, ignore_index=True)

            frame_index += 1
            success, frame = video_fd.read()

        video_fd.release()
    
    train_set_info.to_csv(train_set_info_csv_loc, index=None)
    test_set_info.to_csv(test_set_info_csv_loc, index=None)


def merge_tool_label(train_set_info_csv_loc, test_set_info_csv_loc):
    for csv_info_loc in [train_set_info_csv_loc, test_set_info_csv_loc]:
        csv_data = pd.read_csv(csv_info_loc)
        unique_video_names = csv_data["video_name"].unique().tolist()
        
        merge_csv = pd.DataFrame(None, columns=["video_name", "Frame", "file_loc", "label", "Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"])
        for idx, unique_video_name in enumerate(unique_video_names):
            tool_label_csv = pd.read_csv(os.path.join(tool_label_files, unique_video_name+"-tool.txt"), sep="\t")
            tool_label_csv.insert(loc=0, column="video_name", value=unique_video_name)
            
            # merge
            temp_data = pd.merge(csv_data, tool_label_csv, on=["video_name", "Frame"])
            merge_csv = pd.concat([merge_csv, temp_data])
        
        merge_csv.to_csv(csv_info_loc, index=None)


def add_no_tool_label(train_set_info_csv_loc, test_set_info_csv_loc):
    for csv_info_loc in [train_set_info_csv_loc, test_set_info_csv_loc]:
        csv_data = pd.read_csv(csv_info_loc)
        
        for idx in csv_data.index:
            to_check_row = list(csv_data.loc[idx, ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]])
            csv_data.loc[idx, "NoTool"] = 0 if 1 in to_check_row else 1
    
        csv_data.to_csv(csv_info_loc, index=None)


def parse_args():
    parser = argparse.ArgumentParser(
        usage='python3 make_train_test.py -i path/to/cholec80_dataset -o path/to/output -s 0.5')
    parser.add_argument(
        '-i', '--src_dir', help='path to cholec80 datasets', default=default_src_dir)
    parser.add_argument(
        '-o', '--output_dir', help='path to output', default=default_output_dir)
    parser.add_argument(
        '-t', '--time_step', help='images sampling frequency', default=default_time_step)
    parser.add_argument(
        '-s', '--size', help='ratio of test-set', default=default_test_size)
    parser.add_argument(
        '-x', '--shuffle', help='dataset shuffle', default=default_shuffle)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    # generate a csv contains video name, file_loc and label
    dataset_split(**vars(args))
    
    # merge tool label info into train_set_info.csv or test_set_info.csv
    merge_tool_label(train_set_info_csv_loc, test_set_info_csv_loc)
    
    # add no-tool label info into train_set_info.csv or test_set_info.csv
    add_no_tool_label(train_set_info_csv_loc, test_set_info_csv_loc)
    
