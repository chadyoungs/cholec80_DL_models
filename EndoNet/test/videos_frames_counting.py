import os

import pandas as pd
import cv2

if __name__ == "__main__":
    videos_root_path = r"/media/ExtHDD/cholec80/videos"
    txts_root_path = r"/media/ExtHDD/cholec80/tool_annotations"

    videos_set = os.listdir(videos_root_path)
    txts_set = os.listdir(txts_root_path)

    videos_real_set = [i for i in videos_set if i.__contains__(".mp4")]
    videos_real_set.sort()
    # txts --> tool annotations txts
    txts_real_set = [i for i in txts_set if i.__contains__(".txt")]
    txts_real_set.sort()

    frames_counts = []
    txt_counts = []
    for i, j in zip(videos_real_set, txts_real_set):
        video_path = os.path.join(videos_root_path, i)
        txt_path = os.path.join(txts_root_path, j)

        capture = cv2.VideoCapture(video_path)
        video_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        txt_data = pd.read_csv(txt_path, sep="\t")
        txt_count = txt_data.iloc[-1]["Frame"]
        
        frames_counts.append(video_count)
        txt_counts.append(txt_count)
     
    final_test = pd.DataFrame({"video":frames_counts,
                              "txts":txt_counts})

    print(final_test)
        
       

