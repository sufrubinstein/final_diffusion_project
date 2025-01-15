import torch
import numpy as np
import cv2
import os
from torchvision.utils import make_grid, save_image
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


counter1 = 0


def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=3.3e-6,
                             final_only=False, verbose=False, denoise=True, sigma_noise_0 = 0.1 , save_path = ''):
    images = []
    
    x_0 = x_mod

    global counter1
    counter1 = counter1 + 1
    # save_image(x_mod,'pictures/image_before_{}.png'.format(counter1))

    sigmas = torch.tensor(np.linspace(sigma_noise_0 , sigmas[-1], len(sigmas))).float() #step size sigmas

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2


            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)
                # if(s == 0 and (c >= 10 and c <= 25)):
                    # save_image(grad,'grad_init_output/grad_train{}.png'.format(c))
                    # print(f"grad: {torch.mean(grad.view(grad.shape[0], -1))}")
                    # print(f"x_mod before: {torch.mean(x_mod.view(x_mod.shape[0], -1))}")
                    # mse = x_mod - grad
                    # for i in range(x_mod.size()[0]):
                    #     save_image(x_mod[i],'grad_init_output/mse_image_{}.png'.format(i))

           
                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                # if(s == 0 and c == 9):
                #     save_image(x_mod,'x_mode_before/image_for_sigma_withsc{}.png'.format(c))
                if(sigma == sigmas[0]):
                    continue
                grad_technion = grad + (x_0 - x_mod) / (sigmas[0]**2 - sigma**2)

                # previous eq
                #x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                x_mod = x_mod + step_size * grad_technion + noise * np.sqrt(step_size * 2)
                # if(s == 0 and c == 9):
                #     save_image(x_mod,'x_mode_after/image_for_sigma_withsc{}.png'.format(c))

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
                

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
            
            # print(f"x_mod after: {torch.mean(x_mod.view(x_mod.shape[0], -1))}")

            # if(c == len(sigmas) - 1):
            #     for i in range(x_mod.size()[0]):
            #         save_image(x_mod[i] ,save_path + '/test_sigma_{}_{}.png'.format(sigma_noise_0 ,i))

            if( ((c % 10 == 0) or (c == len(sigmas) - 1))):
                save_image(x_mod ,save_path +'/image_for_sigma{}.png'.format(c))
                if(c == 0):
                    for i in range(x_mod.size()[0]):
                        save_image(x_mod[i] ,save_path + '/noisy_image_{}.png'.format(i))
                for i in range(x_mod.size()[0]):
                    save_image(x_mod[i] ,save_path + '/denoised_image_{}.png'.format(i))


        # save_image(x_mod,'pictures/image_before-denoise_{}.png'.format(counter1))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images


@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, sigmas, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False):
    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))


    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images