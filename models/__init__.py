import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import odak
from tqdm import tqdm
from .unet import *


class holobeam_multiholo(nn.Module):
    """

    Parameters
    ----------
    n_input           : int
                        Number of channels in the input.
    n_hidden          : int
                        Number of channels in the hidden layers.
    n_output          : int
                        Number of channels in the output layer.
    kernel_size       : tuple
                        Kernel size provided as a tuple (e.g., (7, 7).)
    betas             : list
                        Lower and upper bound in variance.
    device            : torch.device
                        Default device is CPU.
    reduction         : str
                        Reduction used for torch.nn.MSELoss and torch.nn.L1Loss. The default is 'sum'.
    """
    def __init__(self,
                 n_input = 1,
                 n_hidden = 16,
                 n_output = 2,
                 kernel_size = (7, 7),
                 device = torch.device('cpu'),
                 reduction = 'sum'
                ):
        super(holobeam_multiholo, self).__init__()
        torch.random.seed()
        self.device = device
        self.reduction = reduction
        self.l2 = torch.nn.MSELoss(reduction = self.reduction)
        self.l1 = torch.nn.L1Loss(reduction = self.reduction)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.network = UNet(
                            dims=self.n_hidden,
                            in_channels=self.n_input,
                            out_channels=self.n_output
                           ).to(self.device)


    def forward(self, x):
        """
        Internal function representing the forward model.
        """
        y = self.network.forward(x) 
        phase_low = y[:, 0].unsqueeze(1)
        phase_high = y[:, 1].unsqueeze(1)
        phase_only = torch.zeros_like(phase_low)
        phase_only[:, :, 0::2, 0::2] = phase_low[:, :,  0::2, 0::2]
        phase_only[:, :, 1::2, 1::2] = phase_low[:, :, 1::2, 1::2]
        phase_only[:, :, 0::2, 1::2] = phase_high[:, :, 0::2, 1::2]
        phase_only[:, :, 1::2, 0::2] = phase_high[:, :, 1::2, 0::2]
        return phase_only


    def evaluate(self, input_data, ground_truth, weights=[1., 0.1]):
        """
        Internal function for evaluating.
        """
        loss = weights[0] * self.l2(input_data, ground_truth) + weights[1] * self.l1(input_data, ground_truth)
        return loss


    def fit(self, dataloader, number_of_epochs=100, learning_rate=1e-5, directory='./output', save_at_every=100):
        """
        Function to train the weights of the multi layer perceptron.

        Parameters
        ----------
        dataloader       : torch.utils.data.DataLoader
                           Data loader.
        number_of_epochs : int
                           Number of epochs.
        learning_rate    : float
                           Learning rate of the optimizer.
        directory        : str
                           Output directory.
        save_at_every    : int
                           Save the model at every given epoch count.
        """
        t_epoch = tqdm(range(number_of_epochs), leave=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for i in t_epoch:
            epoch_loss = 0.
            t_data = tqdm(dataloader, leave=False)
            for j, data in enumerate(t_data):
                self.optimizer.zero_grad()
                images = data[:, 0].unsqueeze(1)
                holograms = data[:, 2].unsqueeze(1)
                estimates = self.forward(images)
                loss = self.evaluate(estimates, holograms)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                description = 'Loss:{:.4f}'.format(loss.item())
                t_data.set_description(description)
                epoch_loss += float(loss.item()) / dataloader.__len__()
            description = 'Epoch Loss:{:.4f}'.format(epoch_loss)
            t_epoch.set_description(description)
            if i % save_at_every == 0:
                self.save_weights(filename='{}/weights_{:04d}.pt'.format(directory, i))
        self.save_weights(filename='{}/weights.pt'.format(directory))
        print(description)

    
    def reconstruct(self,
                    phase,
                    distances=[0.3, -0.3],
                    pixel_pitch=8e-6,
                    wavelength=535e-9,
                    propagation_type='Bandlimited Angular Spectrum'
                   ):
        """
        Function to reconstruct a given hologram.

        Parameters
        ----------
        phase                    : torch.tensor
                                   A phase-only hologram [1x1xmxn].
        distances                : list
                                   Propagation distance in meters.
        pixel_pitch              : float
                                   Pixel pitch in meters.
        wavelength               : float
                                   Wavelength of light.
        propagation_type         : str
                                   Propagation type.

        Returns
        -------
        reconstruction_intensity : torch.tensor
                                   Reconstructed image intensity.
        """
        phase = ((phase[0, 0] / 2.) + 0.5) * (2 * odak.np.pi) % (2 * odak.np.pi)
        hologram = odak.learn.wave.generate_complex_field(
                                                          torch.ones_like(phase),
                                                          phase
                                                         )
        wavenumber = odak.learn.wave.wavenumber(wavelength)
        forward = odak.learn.wave.propagate_beam(
                                                 hologram,
                                                 wavenumber,
                                                 distances[0],
                                                 pixel_pitch,
                                                 wavelength,
                                                 propagation_type=propagation_type,
                                                 zero_padding=[True, False, False]
                                                )
        reconstruction = odak.learn.wave.propagate_beam(
                                                        forward,
                                                        wavenumber,
                                                        distances[1],
                                                        pixel_pitch,
                                                        wavelength,
                                                        propagation_type=propagation_type,
                                                        zero_padding=[False, False, True]
                                                       )
        reconstruction_phase = odak.learn.wave.calculate_phase(reconstruction)
        reconstruction_amplitude = odak.learn.wave.calculate_amplitude(reconstruction)
        reconstruction_intensity = (reconstruction_amplitude)**2
        return reconstruction_intensity, reconstruction_amplitude, reconstruction_phase


    def save_weights(self, filename='./weights.pt'):
        """
        Function to save the current weights of the multi layer perceptron to a file.
        Parameters
        ----------
        filename        : str
                          Filename.
        """
        torch.save(self.network.state_dict(), os.path.expanduser(filename))


    def load_weights(self, filename='./weights.pt'):
        """
        Function to load weights for this multi layer perceptron from a file.
        Parameters
        ----------
        filename        : str
                          Filename.
        """
        self.network.load_state_dict(torch.load(os.path.expanduser(filename)))
        self.network.eval()
