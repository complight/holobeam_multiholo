import sys
import argparse
import torch
import odak
from torch.utils.data import DataLoader

import models
import utils
import time


__title__ = 'Holographic Diffuser'


def main(
        settings_filename='./settings/sample.txt',
        weights_filename=None,
        input_filename=None,
        cuda_device=0,
        mode='train'
        ):
    parser = argparse.ArgumentParser(description=__title__)
    parser.add_argument(
                        '--settings',
                        type=argparse.FileType('r'),
                        help='Filename for the settings file. Default is {}.'.format(settings_filename)
                       )
    parser.add_argument(
                        '--cuda',
                        type=int,
                        help='CUDA device number. Default is {}.'.format(cuda_device)
                       )
    parser.add_argument(
                        '--weights',
                        type=argparse.FileType('r'),
                        help='Filename for the weights file. Default is {}.'.format(settings_filename)
                       )
    parser.add_argument(
                        '--input',
                        type=argparse.FileType('r'),
                        help='Filename for an input data to estimate. Default is {}.'.format(input_filename)
                       )
    args = parser.parse_args()
    if not isinstance(args.settings, type(None)):
        settings_filename = str(args.settings.name)
    if not isinstance(args.weights, type(None)):
        weights_filename = str(args.weights.name)
    if not isinstance(args.input, type(None)):
        input_filename = str(args.input.name)
    if not isinstance(args.cuda, type(None)):
        cuda_device = args.cuda
    settings = odak.tools.load_dictionary(settings_filename)
    device = torch.device(settings["general"]["device"])
    if settings["general"]["device"] == 'cuda':
        torch.cuda.set_device(cuda_device)
    odak.tools.check_directory(settings["general"]["output directory"])
    model = models.holographic_diffuser(
                                        kernel_size=settings["model"]["kernel size"],
                                        n_input=settings["model"]["number of input channels"],
                                        n_hidden=settings["model"]["number of hidden channels"],
                                        n_output=settings["model"]["number of output channels"],
                                        device=device
                                       )
    if not isinstance(weights_filename, type(None)):
        model.load_weights(weights_filename)
    if not isinstance(input_filename, type(None)):
        input_data = torch.load(input_filename).to(device).unsqueeze(0)
        input_data = (input_data - 0.5) * 2
        model_input = torch.zeros(
                                  input_data.shape[0], 
                                  settings["model"]["number of input channels"], 
                                  input_data.shape[-2], 
                                  input_data.shape[-1]
                                 ).to(device)
        model_input[:, 0] = input_data[:, 0]
        odak.learn.tools.save_image('{}/estimate_input.png'.format(settings["general"]["output directory"]),
                                    (model_input[0, 0] / 2.) + 0.5,
                                    cmin=0.,
                                    cmax=1.
                                   )
        t_total = 0
        m = 1 #### SET THIS TO A HIGH NUMBER TO FIND OUT HOW FAST THE METHOD IS
        k = 10
        for i in range(m):
            t_start = time.time_ns()
            torch.no_grad()
            estimate = model.forward(model_input).detach()
            t_finish = time.time_ns()
            t_delta = (t_finish - t_start) / (10 ** 9)
            if i > k:
                print('It took ', t_delta, 'seconds to estimate.')
                t_total += t_delta / (m - k)
        if m > 1:
           print('It took', t_total, 'seconds on average to estimate')
        odak.learn.tools.save_image('{}/estimate_phase.png'.format(settings["general"]["output directory"]), estimate[0, 0], cmin=0., cmax=1.)
        scene_center = settings["hologram"]["delta"] * (settings["hologram"]["number of planes"] - 1) / 2.
        for i in range(settings["hologram"]["number of planes"]):
            distances = settings["hologram"]["distances"].copy()
            distances[1] = distances[1] - scene_center + i * settings["hologram"]["delta"]
            reconstruction_intensity, _, _ = model.reconstruct(
                                                               estimate,
                                                               distances=distances,
                                                               pixel_pitch=settings["hologram"]["pixel pitch"],
                                                               wavelength=settings["hologram"]["wavelength"],
                                                               propagation_type=settings["hologram"]["propagation type"]
                                                              )
            odak.learn.tools.save_image('{}/estimate_reconstruction_{:04d}.png'.format(settings["general"]["output directory"], i), reconstruction_intensity, cmin=0., cmax=1.)
            hologram_phase = input_data[0, 2].unsqueeze(0).unsqueeze(0)
            reconstruction_intensity, _, _ = model.reconstruct(
                                                               hologram_phase,
                                                               distances=distances,
                                                               pixel_pitch=settings["hologram"]["pixel pitch"],
                                                               wavelength=settings["hologram"]["wavelength"],
                                                               propagation_type=settings["hologram"]["propagation type"]
                                                              )
            odak.learn.tools.save_image('{}/ground_truth_reconstruction_{:04d}.png'.format(settings["general"]["output directory"], i), reconstruction_intensity, cmin=0., cmax=1.)

        sys.exit()
    train_dataset = utils.hologram_dataset(
                                           directory=settings['train dataset']['directory'],
                                           device=device
                                          )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    model.fit(
              train_dataloader,
              number_of_epochs=settings["model"]["number of epochs"],
              learning_rate=settings["model"]["learning rate"],
              directory=settings["general"]["output directory"],
              save_at_every=settings["model"]["save at every"]
             )
    model.save_weights(filename='./{}/weights.pt'.format(settings["general"]["output directory"]))


if '__main__' == '__main__':
    sys.exit(main())
