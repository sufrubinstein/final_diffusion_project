import torch
import numpy as np

def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):

    # eps = 1e-8

    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    
    # poisson noise

    # print(f"samples max: {torch.max(samples[0])}")
    # print(f"samples min: {torch.min(samples[0])}")

    # # norm_image = samples / 255.0         #for range [0,1]
    # # norm_image = samples    #for range [0,255]
    sqrt_images = torch.sqrt(samples) 
    sigma_start_noise = used_sigmas
    poisson_noise = torch.randn(sqrt_images.size()).to(samples.device) * sigma_start_noise #define gaussian noise 
                                                                                            #for poisson noise case
    noisy_init_samples = sqrt_images + poisson_noise
    noisy_init_samples = noisy_init_samples * sqrt_images

    perturbed_samples = noisy_init_samples

    target = (samples - perturbed_samples) / (used_sigmas ** 2)

    # print(f"traget before: {torch.mean(target.view(target.shape[0], -1))}")

    # for poisson noise
    target = target / torch.clamp(samples, min= 1 / 255.0)

    # target = target.view(target.shape[0], -1)
    
    # print(f"traget after: {torch.mean(target)}")

    # print(f"samples: {torch.mean(samples.view(samples.shape[0], -1))}")

    # gaussian noise

    # noise = torch.randn_like(samples) * used_sigmas
    # perturbed_samples = samples + noise
    # target = - 1 / (used_sigmas ** 2) * noise

    scores = scorenet(perturbed_samples, labels)

    target = target.view(target.shape[0], -1)

    scores = scores.view(scores.shape[0], -1)

    # print(f"scores: {torch.mean(scores)}")

    
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)
