
import utils
import argparse
import os
import display_digit as display_results

import model_vae.one_label_data_loader as data_loader
import model_vae.vae_net as vae_net

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/check-data-loader', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/check_data_loading', help="")

args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

dataloader = data_loader.fetch_dataloader(['dev'], args.data_dir, params)
train_dl = dataloader['dev']

fig = display_results.create_figure()
epoch = 1

for i, (train_batch, labels_batch) in enumerate(train_dl):
    train_batch_shaped = vae_net.vectors_to_samples(train_batch)
    labels_batch_shaped = vae_net.labels_to_titles(labels_batch)
    display_results.fill_figure(train_batch_shaped, fig, epoch + 1, args.model_dir, 'data', labels_batch_shaped)
