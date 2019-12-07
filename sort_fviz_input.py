import os
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="toSync/Thesis/DL-Pytorch-data/to_shap", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='data/gest_laws/similarity/few-digits-RGB-change/segs-5-7/size-100/digit-1/data.csv', help="Directory containing the dataset")
parser.add_argument('--out_data_dir', default='data/gest_laws/similarity/few-digits-RGB-change/segs-5-7/size-100/digit-1/train/data-sorted.csv', help="Directory containing the dataset")

if __name__ == '__main__':
    num_colors = 5
    num_runs = 20
    num_conds = 4
    args = parser.parse_args()
    if args.parent_dir:
        past_to_drive = os.environ['OneDrive']
        os.chdir(os.path.join(past_to_drive, args.parent_dir))
        # print(os.getcwd())

    data_frame = pd.read_csv(args.data_dir, header=None)

    # shuffled_data_frame = data_frame  # data_frame.sample(n=25)
    # shuffled_data_frame = data_frame.sample(frac=1)

    data = data_frame.values
    data = np.float32(data)

    images = data[:, :24]

    # print(images.shape)
    temp = np.arange(images.shape[0])
    # print(temp % (4*num_colors))
    # print(temp // (4*num_colors))

    sorted_data = np.zeros(data.shape)
    ind_orig = []
    ind_sorted = []

    for i in range(num_runs):
        for k in range(num_conds):
            for j in range(num_colors):
                sorted_data[num_conds*num_colors*i + num_colors*k + j] = data[num_conds*k + i*num_runs + j % num_colors]  # [num_colors*num_conds*k + i*num_colors + j]
                ind_orig.append(num_conds*k + i*num_runs + j % num_colors)
                ind_sorted.append(num_conds*num_colors*i + num_colors*k + j)

    print(ind_orig)
    print('****************')
    print(ind_sorted)
    print('****************')

    # sorted_df = pd.DataFrame(sorted_data)
    # sorted_df.to_csv(args.out_data_dir, index=False)
