import os
import sys
import numpy as np

def overlapped(path_to_videos):
    """
    path_to_videos: such that path_to_videos/s{i}/abcd1e/0**.png
    """
    num_test, num_val = 128, 128
    train_list, val_list, test_list = [], [], []
    
    speakers = os.listdir(path_to_videos)

    for spk in speakers:
        video_dirs = os.listdir(os.path.join(path_to_videos, spk))
        np.random.shuffle(video_dirs)

        for vid_dir in video_dirs[:num_test]:
            test_list.append(os.path.join(spk, vid_dir)+'\n')

        for vid_dir in video_dirs[num_test : num_test + num_val]:
            val_list.append(os.path.join(spk, vid_dir)+'\n')

        for vid_dir in video_dirs[num_test + num_val:]:
            train_list.append(os.path.join(spk, vid_dir)+'\n')

    if not os.path.exists("overlapped"):
        os.makedirs("overlapped")

    with open(os.path.join('overlapped', 'train_dirs.txt'), 'w') as f:
        f.writelines(train_list)

    with open(os.path.join('overlapped', 'val_dirs.txt'), 'w') as f:
        f.writelines(val_list)

    with open(os.path.join('overlapped', 'test_dirs.txt'), 'w') as f:
        f.writelines(test_list)

    return 1

def unseen(path_to_videos, test_spks):
    """
    path_to_videos: such that path_to_videos/s{i}/abcd1e/0**.img
    """
    num_val = 200

    train_list, val_list, test_list = [], [], []
    speakers = os.listdir(path_to_videos)

    for spk in speakers:

        video_dirs = os.listdir(os.path.join(path_to_videos, spk))

        if spk not in test_spks:
        
            np.random.shuffle(video_dirs)

            for vid_dir in video_dirs[:num_val]:
                val_list.append(os.path.join(spk, vid_dir)+'\n')

            for vid_dir in video_dirs[num_val:]:
                train_list.append(os.path.join(spk, vid_dir)+'\n')

        else:

            for vid_dir in video_dirs:
                test_list.append(os.path.join(spk, vid_dir)+'\n')

    if not os.path.exists("unseen"):
        os.makedirs("unseen")

    with open(os.path.join('unseen', 'train_dirs.txt'), 'w') as f:
        f.writelines(train_list)

    with open(os.path.join('unseen', 'val_dirs.txt'), 'w') as f:
        f.writelines(val_list)

    with open(os.path.join('unseen', 'test_dirs.txt'), 'w') as f:
        f.writelines(test_list)

    return 1

if __name__ == '__main__':
    path_to_videos = sys.argv[1]
    overlapped_unseen = sys.argv[2]

    if overlapped_unseen == 'overlapped':
        overlapped(path_to_videos)

    elif overlapped_unseen == 'unseen':
        test_spk_nums = [1, 2, 20, 22]
        test_spks = [f"s{i}" for i in test_spk_nums]
        unseen(path_to_videos, test_spks)

    else:
        print('Second argument should be \"overlapped\" or \"unseen\"')