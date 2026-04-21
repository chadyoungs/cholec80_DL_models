import os

import numpy as np
import pandas as pd

import cv2

root = r"/media/ExtHDD/cholec80"
videos_path = os.path.join(root, "videos")
labels_by_frame_path = os.path.join(root, "phase_annotations")

class TrainTestOptError(Exception):
    """
    Train Test Option Setting Error
    """
    
def get_img_width_height():
    capture = cv2.VideoCapture(os.path.join(videos_path, "video01.mp4"))
    
    print("width", int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), "\n",
          "height", int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

def get_video_len(video_root, video):
    capture = cv2.VideoCapture(os.path.join(video_root, video))
    
    return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

def get_dataframe_len(txt_root, txt):
    txt_data = pd.read_csv(os.path.join(txt_root, txt), sep="\t")
    
    return len(txt_data)
       
def label_check():
    """
    To check the annotations
    """
    # get videos
    videos = [video for video in os.listdir(videos_path) if video.__contains__(".mp4")]
    videos.sort()
    
    # get labels_by_timestamp txts
    txts_by_timestamp = [txt for txt in os.listdir(videos_path) if txt.__contains__(".txt")]
    txts_by_timestamp.sort()
    
    # get labels_by_frame txts
    txts_by_frame = [txt for txt in os.listdir(labels_by_frame_path) if txt.__contains__(".txt")]
    txts_by_frame.sort()
    
    check_data = pd.DataFrame(None, columns=["video_len", "txt_by_timestamp_len", "txt_by_frame_len"])
    
    for video, txt_by_timestamp, txt_by_frame in zip(videos, txts_by_timestamp, txts_by_frame):
        video_name_f = video.rsplit(".")[0]
        video_name_s = txt_by_timestamp.rsplit("-")[0]
        video_name_t = txt_by_frame.rsplit("-")[0]
        
        assert video_name_f == video_name_s == video_name_t
        
        pd_series = pd.Series({"video_len": get_video_len(videos_path, video), "txt_by_timestamp_len": get_dataframe_len(videos_path, txt_by_timestamp), 
                                        "txt_by_frame_len": get_dataframe_len(labels_by_frame_path, txt_by_frame)}, name=video_name_f)
        check_data = check_data.append(pd_series)
    
    check_data.to_csv(os.path.join(os.path.dirname(__file__), "dataset", "label_check.csv"))
        
def plotting(info):
    for loss, score1, score2 in zip(['train_losses', 'test_loss'], ['train_scores1', 'test_score1'], ['train_scores2', 'test_score2']):
        y1 = info[loss]
        y2 = info[score1]
        y3 = info[score2]

        plt.plot(np.linspace(1, len(y1), 1), y1, y2, y3)
    pass

if __name__ == "__main__":
    #label_check()
    get_img_width_height()
        
       

