"""Basic models."""
from tqdm import tqdm

import torch
from torch import nn
import torchvision as tv
from torchvision import transforms


def generator_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """
    Function for returning a block of the generator's neural network given input and output dimensions.
    Args:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar

    Returns:
        A generator neural network layer, with a linear transformation followed by a batch normalization and then a relu
        activation
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )


def discriminator_block(input_dim: int, output_dim: int) -> nn.Sequential:
    """
    Function for returning a neural network of the discriminator given input and output dimensions.

    Args:
        input_dim: the dimension of the input vector
        output_dim: the dimension of the output vector

    Returns:
        A discriminator neural network layer, with a linear transformation followed by an `nn.LeakyReLU` activation with
        negative slope of 0.2 (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html).
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2)
    )


def get_noise(n_samples: int, z_dim: int, device: str = 'cpu'):
    """
     Function for creating noise vectors: Given the dimensions (n_samples, z_dim).

    Args:
        n_samples: the number of samples to generate
        z_dim: the dimension of the noise vector
        device: the device type

    Returns:
        A noise tensor of dimensions (n_samples, z_dim) filled with random numbers from the normal distribution.
    """
    return torch.randn(n_samples, z_dim, device=device)


class Generator(nn.Module):
    """Generator Class."""

    def __init__(self, z_dim: int = 10, im_dim: int = 784, hidden_dim: int = 128) -> None:
        super(Generator, self).__init__()
        """
        The Generator Class, which takes a noise vector as input and produces an image as output.
        
        Args:
            z_dim: the dimension of the noise vector
            im_dim: the dimension of the images, fitted for the dataset used. MNIST images are 28 x 28 = 784 so that 
                is your default
            hidden_dim: the inner dimension
        """
        self.gen = nn.Sequential(
            generator_block(z_dim, hidden_dim),
            generator_block(hidden_dim, hidden_dim * 2),
            generator_block(hidden_dim * 2, hidden_dim * 4),
            generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Function for completing a forward pass of the generator: Given a noise tensor.
        Args:

            noise: a noise tensor with dimensions (n_samples, z_dim)

        Returns:
            The generated images.
        """
        return self.gen(noise)

    def get_gen(self) -> nn.Sequential:
        """
        Retrieves the sequential model.

        Returns:
            The sequential model.
        """
        return self.gen


class Discriminator(nn.Module):
    """Discriminator Class."""

    def __init__(self, im_dim: int = 784, hidden_dim: int = 128) -> None:
        super(Discriminator, self).__init__()
        """
        The Discriminator Class, which takes a noise vector as input and produces an image as output.

        Args:
            im_dim: the dimension of the images, fitted for the dataset used. MNIST images are 28 x 28 = 784 so that 
                is your default
            hidden_dim: the inner dimension
        """
        self.disc = nn.Sequential(
            discriminator_block(im_dim, hidden_dim * 4),
            discriminator_block(hidden_dim * 4, hidden_dim * 2),
            discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Function for completing a forward pass of the discriminator: Given a noise tensor.
        Args:

            image: an image tensor with dimensions (n_samples, im_dim)

        Returns:
            The generated images.
        """
        return self.disc(image)

    def get_disc(self) -> nn.Sequential:
        """
        Retrieves the sequential model.

        Returns:
            The sequential model.
        """
        return self.disc


def get_disc_loss(
        gen: Generator, disc: Discriminator, criterion: nn.BCEWithLogitsLoss, real, num_images: int, z_dim: int,
        device: str
) -> torch.Tensor:
    """
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the images
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce,
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    """
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


def get_gen_loss(gen: Generator, disc: Discriminator, criterion: nn.BCEWithLogitsLoss, num_images: int, z_dim: int,
             device: str) -> torch.Tensor:
    """
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare
               the discriminator's predictions to the ground truth reality of the images
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce,
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    """
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss


def main() -> None:
    """Function for setting up the training environment."""
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 10
    z_dim = 64
    display_step = 1000
    batch_size = 128
    lr = 0.00001
    device = 'cpu'

    data_loader = torch.utils.data.DataLoader(
        tv.datasets.MNIST('.', download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True
    )

    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    test_generator = True  # Whether the generator should be tested
    gen_loss = False
    error = False

    for epoch in range(n_epochs):
        # Dataloader returns the batches
        for real, _ in tqdm(data_loader):
            cur_batch_size = len(real)

            # Flatten the batch of real images from the dataset
            real = real.view(cur_batch_size, -1).to(device)

            # Zero out the gradients before backpropagation
            disc_opt.zero_grad()

            # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            disc_opt.step()

            # For testing purposes, to keep track of the generator weights
            if test_generator:
                old_generator_weights = gen.gen[0][0].weight.detach().clone()

            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward()
            gen_opt.step()

            # For testing purposes, to check that your code changes the generator weights
            if test_generator:
                try:
                    assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Runtime tests have failed")

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(cur_batch_size, z_dim, device=device
                                       )
                fake = gen(fake_noise)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1


if __name__ == '__main__':
    main()
