import argparse
import numpy as np
import os
import pandas as pd
import torch


import plot_digit as display_results

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="toSync/Thesis/DL-Pytorch-data", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir',
                    default='experiments/acgan_model/hiding_scheme/partial_class/hard/partial_class_hard_hidden_2/three_layers_exact_net/hidden_size_300/to-thesis/samples_epoch_380.csv',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/acgan_model/hiding_scheme/partial_class/hard/partial_class_hard_hidden_2/three_layers_exact_net/hidden_size_300/to-thesis',
                    help="Directory containing ACGAN output")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'



# default='experiments/acgan_model/hiding_scheme/partial_class/hard/combined-samples-hard-for-plot.csv'
if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    if args.parent_dir and not torch.cuda.is_available():
        past_to_drive = os.environ['OneDrive']
        os.chdir(os.path.join(past_to_drive, args.parent_dir))
    # json_path = os.path.join(args.model_dir, 'params.json')
    # assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # params = utils.Params(json_path)

    data_frame = pd.read_csv(args.data_dir, header=None)
    # temp = np.float32(data_frame.values)
    # print(temp[0])
    shuffled_data_frame = data_frame  #data_frame.sample(n=25)
    # shuffled_data_frame = data_frame.sample(frac=1)

    images = shuffled_data_frame.values
    images = np.float32(images)
    # print(images.shape)

    gray_images = display_results.convert_digit_to_after_filer_grayscale(images)
    # print(images[0])

    tensor_image = torch.from_numpy(images)
    tensor_image = tensor_image.type(torch.FloatTensor)

    test_samples_reshaped = display_results.vectors_to_samples(tensor_image)
    # print(np.array(test_samples_reshaped).shape)

    gray_tensor_image = torch.from_numpy(gray_images)
    gray_tensor_image = gray_tensor_image.type(torch.FloatTensor)

    gray_test_samples_reshaped = display_results.vectors_to_samples(gray_tensor_image)
    # print(np.array(gray_test_samples_reshaped).shape)

    fig = display_results.create_figure()

    display_results.feed_digits_to_figure(test_samples_reshaped, fig, args.model_dir, 0, 255, dtype='num_380_masking')
    display_results.feed_digits_to_figure(gray_test_samples_reshaped, fig, args.model_dir, 0, 255, dtype='num_380_hidden')  # 'DO_hidden')
    # display_results.fill_figure(test_samples_reshaped, fig, args.model_dir, 0, 255, dtype='A&D_masking')
    # display_results.fill_figure(gray_test_samples_reshaped, fig, args.model_dir, 0, 255, dtype='A&D_hidden')

    display_results.close_figure(fig)

