import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--fold_dir', default='toSync/Thesis/DL-Pytorch-data/k-fold-cross-val/train-and-eval/4-fold/', help="Directory containing the classification results")
parser.add_argument('--test_folder', default='toSync/Thesis/DL-Pytorch-data/data/with-grayscale/k-fold-cross-val/4-fold/test/',
                    help="path to test file")
parser.add_argument('--re_save', default=False, type=bool)

# left column is output, right is ground truth


def find_row_index(query, table):
    return (query == table).all(axis=1).nonzero()[0][0]


if __name__ == '__main__':
    drive_path = os.environ['OneDrive']
    args = parser.parse_args()
    folder_path = os.path.join(os.path.expanduser(drive_path), args.fold_dir)

    # read files for classification result and ground truth
    incorrect_filename = 'evaluate_test_incorrect_samples.csv'
    correct_filename = 'evaluate_test_correct_samples.csv'
    files_to_read = [incorrect_filename, correct_filename]
    net_out = []
    ground_truth = []
    classes = [i for i in range(10)]
    for file_name in files_to_read:
        data_path = os.path.join(folder_path, file_name)
        file = pd.read_csv(data_path, header=None)
        numpy_vals = file.values
        net_out.extend(numpy_vals[:, -2].tolist())
        ground_truth.extend(numpy_vals[:, -1].tolist())

    # np.set_printoptions(precision=2)
    res = classification_report(ground_truth, net_out)
    if args.re_save is True:
        txt_file_path = os.path.join(folder_path, 'classification_report.txt')
        with open(txt_file_path, "w") as text_file:
            text_file.write(res)

    res_to_file = classification_report(ground_truth, net_out, output_dict=True)
    df_res = pd.DataFrame.from_dict(res_to_file).round(4)
    ind = [str(i) for i in range(10)]
    df_res_classes = df_res[ind].T

    cols = df_res_classes.columns.tolist()
    new_cols = [cols[1], cols[2], cols[0], cols[3]]
    df_res_classes = df_res_classes[new_cols]
    if args.re_save is True:
        df_res_classes.to_csv(os.path.join(folder_path, 'classification_report.csv'))

    conf_m = confusion_matrix(ground_truth, net_out, labels=classes, normalize='true')
    df_cm = pd.DataFrame(conf_m, index=classes, columns=classes)
    plt.figure()
    ax = plt.gca()
    fig = plt.gcf()
    sn.heatmap(df_cm, vmin=0, vmax=1, cmap=plt.cm.Greens, linecolor='black', linewidths=0.5, annot=True,
               annot_kws={"fontsize": 9})
    plt.xlabel('network output', labelpad=10)
    plt.ylabel('ground truth', labelpad=10)
    ax.set_ylim([10.1, -0.1])
    ax.set_xlim([-0.1, 10.1])
    im_path = os.path.join(os.path.expanduser(drive_path), args.fold_dir, 'confusion_matrix.png')
    if args.re_save is True:
        fig.savefig(im_path, bbox_inches='tight')

    # find accuracy per file type ##################
    t_inp_file = 'synthetic-gray-data-test.csv'
    inc_out_file = 'evaluate_test_incorrect_samples_w_indices.csv'
    # read full file
    test_data_path = os.path.join(drive_path, args.test_folder, t_inp_file)
    test_file = pd.read_csv(test_data_path, header=None).values
    test_file = test_file[:,:-1]

    # read query file
    query_data_path = os.path.join(folder_path, incorrect_filename)
    orig_query_file = pd.read_csv(query_data_path, header=None).values
    query_file = orig_query_file[:, :-2]

    indexed_query_file = np.zeros((np.shape(orig_query_file)[0], np.shape(orig_query_file)[1]+1), dtype=int)
    indexed_query_file[:, :-1] = orig_query_file
    # indexed_query_file = query_file

    for line_ind in range(np.shape(query_file)[0]):
        qline = query_file[line_ind]
        ind_in_test = find_row_index(qline, test_file)
        indexed_query_file[line_ind, -1] = ind_in_test

    # out_df = pd.DataFrame(indexed_query_file, header=None)
    # out_df.to_csv(os.path.join(folder_path, inc_out_file))
    np.savetxt(os.path.join(folder_path, inc_out_file), indexed_query_file, delimiter=",", fmt='%d')
