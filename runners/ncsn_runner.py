import numpy as np
import glob
import tqdm
from losses.dsm import anneal_dsm_score_estimation
import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from models.ncsn import NCSN, NCSNdeeper
from datasets import get_dataset, data_transform, inverse_data_transform
from losses import get_optimizer
from models import (anneal_Langevin_dynamics,
                    anneal_Langevin_dynamics_inpainting,
                    anneal_Langevin_dynamics_interpolation)
from models import get_sigmas
from models.ema import EMAHelper

from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr , structural_similarity as ssim
# from our_functions.psnr_func import calculate_fid

__all__ = ['NCSNRunner']

def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)

class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def train(self):
        dataset, test_dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)

        test_iter = iter(test_loader)
        

        # # try for random images in test and not the same image
        # images_test, labels_test = next(test_iter)
        # save_image(images_test,'2_test_train_1/original_images.png')
        # for i in range(images_test.size()[0]):
        #     save_image(images_test[i],'2_test_train_1/original_image_{}.png'.format(i))

        psnr_arr = []
        fid_arr = []
        ssim_arr = []
        
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = get_model(self.config)

        score = torch.nn.DataParallel(score)
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        sigmas = get_sigmas(self.config) #training sigmas 90-0.01 for celeba
        # print(sigmas)

        if self.config.training.log_all_sigmas:
            ### Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(len(sigmas))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(len(sigmas)):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(len(sigmas)):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                score.train()
                step += 1

                X = X.to(self.config.device)
                X = data_transform(self.config, X)

                loss = anneal_dsm_score_estimation(score, X, sigmas, None,
                                                   self.config.training.anneal_power,
                                                   hook)
                tb_logger.add_scalar('loss', loss, global_step=step)
                tb_hook()

                logging.info("step: {}, loss: {}".format(step, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0

                # if step % 100 == 0: #Original code
                if step % self.config.training.snapshot_freq == 0: #for testing all test images - for psnr calculations
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    save_path = '0625_train_1'

                    test_score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                        save_image(test_X, save_path + '/original_images.png')
                        for i in range(test_X.size()[0]):
                            save_image(test_X[i],save_path + '/original_image_{}.png'.format(i))
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)
                        save_image(test_X,save_path + '/original_images.png')
                        for i in range(test_X.size()[0]):
                            save_image(test_X[i],save_path + '/original_image_{}.png'.format(i))

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(test_score, test_X, sigmas, None,
                                                                    self.config.training.anneal_power,
                                                                    hook=test_hook)
                        tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
                        test_tb_hook()
                        logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))

                        del test_score

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    # torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()

                        ## Different part from NeurIPS 2019.
                        ## Random state will be affected because of sampling during training time.


                        # ORIGINAL CODE!!!!

                        # init_samples = torch.rand(36, self.config.data.channels,
                        #                           self.config.data.image_size, self.config.data.image_size,
                        #                           device=self.config.device)
                        # init_samples = data_transform(self.config, init_samples)


                        # test_pictures = test_X[:36]
                        # save_image(test_pictures,'picturetest/test_clean_images_{}.png'.format(step))
                        # test_pictures = test_pictures + torch.randn_like(test_pictures)*0.25
                        # save_image(test_pictures,'picturetest/test_noisy_images_{}.png'.format(step))

                        # init_samples = data_transform(self.config, test_pictures)




                        # Load the image
                        # image_path = 'real_test/original_images.png'
                        # image = Image.open(image_path)

                        # image = images_test.to(self.config.device)                    ###### OLD CODE FOR FIRST 8 IMAGES ALWAYS
                        image = test_X.to(self.config.device)                           ###### NEW CODE FOR ALL IMAGES TESTING

                        # print(f"traget after: {torch.max(image[0])}")
                        # print(f"traget after: {torch.min(image[0])}")

                        # plt.hist(image[0].view(-1).cpu().numpy(), bins=100)
                        # plt.savefig('test/hist.png')
                        
                        # Define the transform to resize and normalize the image
                        transform = transforms.Compose([
                            transforms.Resize((self.config.data.image_size, self.config.data.image_size)),
                            transforms.ToTensor()
                        ])

# possible poisson transformation

# no dividing by 255

                        # norm_image = image / 255.0
                        # for i in range(norm_image.size()[0]):
                        #     save_image(norm_image[i] * 255.0 ,'grad_init_output/norm_image_{}.png'.format(i))

                        # image = image * 255.0
                        # print(f"max range after multiply by 255: {torch.max(image[0])}")
                        # print(f"min range after multiply by 255: {torch.min(image[0])}")
                        sqrt_images = torch.sqrt(image)
                        sigma_start_noise =  1 / np.sqrt(255)
                        poisson_noise = torch.randn(sqrt_images.size()).to(self.config.device) * sigma_start_noise #define gaussian noise 
                                                                                                                    #for poisson noise case
                        noisy_init_samples = sqrt_images + poisson_noise
                        noisy_init_samples = noisy_init_samples * sqrt_images
                        # print("Min value of noisy_init_samples:", torch.min(noisy_init_samples))
                        # temp_noisy_image = (noisy_init_samples - torch.min(noisy_init_samples)) / (torch.max(noisy_init_samples) - torch.min(noisy_init_samples))
                        
                        # for i in range(temp_noisy_image.size()[0]):
                        #    save_image(temp_noisy_image[i] ,save_path + '/temp_noisy_possion_image_{}.png'.format(i))


                        # # #Anscombe transformation
                        # noisy_init_samples = 2 * torch.sqrt(torch.clamp(noisy_init_samples + 3.0 / 8.0, min=0))
                        # temp_max_anscombe = torch.max(noisy_init_samples)
                        # print(f"maximal value after taking root of the image: {temp_max_anscombe}")
                        # noisy_init_samples = noisy_init_samples / temp_max_anscombe #[0,1] normalization
                        # print(f"image max value after anscombe tranform: {torch.max(noisy_init_samples[0])}")
                        # print(f"image min value after anscombe tranform: {torch.min(noisy_init_samples[0])}")
                        # for i in range(noisy_init_samples.size()[0]):
                        #    save_image(noisy_init_samples[i] ,save_path + '/image_after_anscombe_tranform_{}.png'.format(i))
                        # sigma_start_noise = 1.0 / temp_max_anscombe
                        # print("sigma start noise value:",sigma_start_noise)



## maybe we need to normilize the outcome back to [0,1], right now there might be values greater than 1 in the image.

                       
                        # # Adding Gaussian noise
                        # sigma_start_noise = 0.702
                    

                        # gaussian_noise = torch.randn(image.size()).to(self.config.device) * sigma_start_noise

                        # # Add Gaussian noise to the image tensor, then clamp between 0 and 1
                        # noisy_init_samples = torch.clamp(image + gaussian_noise, 0, 1)

                        init_samples = noisy_init_samples

                        # Repeat the image tensor to match the batch size and channels
                        # init_samples = noisy_init_samples.unsqueeze(0).repeat(1, 1, 1,1).to(self.config.device)
                        init_samples = data_transform(self.config, init_samples)
                        
                        
                        # here we changed from init_samples to noisy_image_samples!!!! 
                        all_samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                               self.config.sampling.n_steps_each,
                                                               self.config.sampling.step_lr,
                                                               final_only=True, verbose=False,
                                                               denoise=self.config.sampling.denoise, 
                                                               sigma_noise_0 = sigma_start_noise, # sigma_start
                                                               save_path = save_path)
                        
                        # sigma_check = np.linspace(1, 0.01, 40)
                        # for i in range(40):
                        #     all_samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                        #                                        self.config.sampling.n_steps_each,
                        #                                        self.config.sampling.step_lr,
                        #                                        final_only=True, verbose=False,
                        #                                        denoise=self.config.sampling.denoise, 
                        #                                        sigma_noise_0 = sigma_check[i], # sigma_start
                        #                                        save_path = save_path)



                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, 6)
                        save_image(image_grid,
                                   os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))
                        torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))
                        
                        
                        # denoised_images_anscombe = all_samples[-1]
                        # torch.clamp(denoised_images_anscombe, 0, 1)
                        # denoised_images_anscombe = denoised_images_anscombe * 35.37750244140625 # temp_max_anscombe
                        # denoised_image_before_inverse_anscombe = (denoised_images_anscombe - torch.min(denoised_images_anscombe)) / (torch.max(denoised_images_anscombe) - torch.min(denoised_images_anscombe))
                        # save_image(denoised_image_before_inverse_anscombe ,save_path +'/denoised_image_before_inverse_anscombe.png')
                        # inverse_anscombe = (denoised_images_anscombe / 2) ** 2 - 3.0 / 8.0
                        # inverse_anscombe = (inverse_anscombe - torch.min(inverse_anscombe)) / (torch.max(inverse_anscombe) - torch.min(inverse_anscombe))

                        # print(f"image max value after inverse anscombe tranform: {torch.max(inverse_anscombe[0])}")
                        # print(f"image min value after inverse anscombe tranform: {torch.min(inverse_anscombe[0])}")
                        # for i in range(inverse_anscombe.size()[0]):
                        #    save_image(inverse_anscombe[i] ,save_path + '/denoised_image_after_inverse_anscombe_{}.png'.format(i))

                        # for i in range(8):

                        #     # Load the images before and after denoising
                        #     original_img = Image.open(save_path + "/original_image_{}.png".format(i))
                        #     denoised_img = Image.open(save_path + "/denoised_image_{}.png".format(i))

                        #     # Resize the original image to 128x128 to match the denoised image
                        #     original_img_resized = original_img.resize((128, 128), Image.LANCZOS)

                        #     # Convert both images to RGB to ensure they have 3 channels
                        #     original_img_rgb = original_img_resized.convert('RGB')
                        #     denoised_img_rgb = denoised_img.convert('RGB')

                        #     original_img_rgb_array = np.array(original_img_rgb)
                        #     denoised_img_rgb_array = np.array(denoised_img_rgb)

                        #     # print(np.max(original_img_rgb_array))
                        #     # print(np.min(original_img_rgb_array))
                        #     # print(np.max(denoised_img_rgb_array))
                        #     # print(np.min(denoised_img_rgb_array))

                        #     # Calculate PSNR
                        #     psnr_value = psnr(original_img_rgb_array, denoised_img_rgb_array)
                        #     fid_value = calculate_fid(original_img_rgb_array , denoised_img_rgb_array)
                        #     ssim_value = ssim(original_img_rgb_array, denoised_img_rgb_array , channel_axis=2)
                            
                        #     psnr_arr.append(psnr_value)
                        #     fid_arr.append(fid_value)
                        #     ssim_arr.append(ssim_value)


                        # #print PSNR
                        # print(f"PSNR values are: {np.round(psnr_arr,2)} dB", flush=True)
                        # print(f"PSNR mean is: {np.mean(psnr_arr):.2f} dB", flush=True)
                        # print(f"PSNR std is: {np.std(psnr_arr):.2f} dB", flush=True)

                        # #print FID
                        # # print(f"FID values are: {np.round(fid_arr,2)}", flush=True)
                        # print(f"FID mean is: {np.mean(fid_arr):.2f}", flush=True)
                        # print(f"FID std is: {np.std(fid_arr):.2f}", flush=True)

                        # #print SSIM
                        # # print(f"SSIM values are: {np.round(ssim_arr,2)}", flush=True)
                        # print(f"SSIM mean is: {np.mean(ssim_arr):.2f}", flush=True)
                        # print(f"SSIM std is: {np.std(ssim_arr):.2f}", flush=True)


                        del test_score
                        del all_samples

    def sample(self):
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        dataset, _ = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                num_workers=4)

        score.eval()

        if not self.config.sampling.fid:
            if self.config.sampling.inpainting:
                data_iter = iter(dataloader)
                refer_images, _ = next(data_iter)
                refer_images = refer_images.to(self.config.device)
                width = int(np.sqrt(self.config.sampling.batch_size))
                init_samples = torch.rand(width, width, self.config.data.channels,
                                          self.config.data.image_size,
                                          self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)
                all_samples = anneal_Langevin_dynamics_inpainting(init_samples, refer_images[:width, ...], score,
                                                                  sigmas,
                                                                  self.config.data.image_size,
                                                                  self.config.sampling.n_steps_each,
                                                                  self.config.sampling.step_lr)

                torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pth'))
                refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1,
                                                                                                     *refer_images.shape[
                                                                                                      1:])
                save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'), nrow=width)

                if not self.config.sampling.final_only:
                    for i, sample in enumerate(tqdm.tqdm(all_samples)):
                        sample = sample.view(self.config.sampling.batch_size, self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'completion_{}.pth'.format(i)))
                else:
                    sample = all_samples[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    'completion_{}.pth'.format(self.config.sampling.ckpt_id)))

            elif self.config.sampling.interpolation:
                if self.config.sampling.data_init:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                else:
                    init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_size,
                                              device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics_interpolation(init_samples, score, sigmas,
                                                                     self.config.sampling.n_interpolations,
                                                                     self.config.sampling.n_steps_each,
                                                                     self.config.sampling.step_lr, verbose=True,
                                                                     final_only=self.config.sampling.final_only)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, nrow=self.config.sampling.n_interpolations)
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
                else:
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, self.config.sampling.n_interpolations)
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    'samples_{}.pth'.format(self.config.sampling.ckpt_id)))

            else:
                if self.config.sampling.data_init:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                else:
                    init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_size,
                                              device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=True,
                                                       final_only=self.config.sampling.final_only,
                                                       denoise=self.config.sampling.denoise)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
                else:
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    'samples_{}.pth'.format(self.config.sampling.ckpt_id)))

        else:
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // self.config.sampling.batch_size
            if self.config.sampling.data_init:
                dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                        num_workers=4)
                data_iter = iter(dataloader)

            img_id = 0
            for _ in tqdm.tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
                if self.config.sampling.data_init:
                    try:
                        samples, _ = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    samples = samples + sigmas_th[0] * torch.randn_like(samples)
                else:
                    samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size, device=self.config.device)
                    samples = data_transform(self.config, samples)

                all_samples = anneal_Langevin_dynamics(samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=False,
                                                       denoise=self.config.sampling.denoise)

                samples = all_samples[-1]
                for img in samples:
                    img = inverse_data_transform(self.config, img)

                    save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
                    img_id += 1

    def test(self):
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas = get_sigmas(self.config)

        dataset, test_dataset = get_dataset(self.args, self.config)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.test.batch_size, shuffle=True,
                                     num_workers=self.config.data.num_workers, drop_last=True)

        verbose = False
        for ckpt in tqdm.tqdm(range(self.config.test.begin_ckpt, self.config.test.end_ckpt + 1, 5000),
                              desc="processing ckpt:"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            step = 0
            mean_loss = 0.
            mean_grad_norm = 0.
            average_grad_scale = 0.
            for x, y in test_dataloader:
                step += 1

                x = x.to(self.config.device)
                x = data_transform(self.config, x)

                with torch.no_grad():
                    test_loss = anneal_dsm_score_estimation(score, x, sigmas, None,
                                                            self.config.training.anneal_power)
                    if verbose:
                        logging.info("step: {}, test_loss: {}".format(step, test_loss.item()))

                    mean_loss += test_loss.item()

            mean_loss /= step
            mean_grad_norm /= step
            average_grad_scale /= step

            logging.info("ckpt: {}, average test loss: {}".format(
                ckpt, mean_loss
            ))

    def fast_fid(self):
        ### Test the fids of ensembled checkpoints.
        ### Shouldn't be used for models with ema
        if self.config.fast_fid.ensemble:
            if self.config.model.ema:
                raise RuntimeError("Cannot apply ensembling to models with EMA.")
            self.fast_ensemble_fid()
            return

        from evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        for ckpt in tqdm.tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def fast_ensemble_fid(self):
        from evaluation.fid_score import get_fid, get_fid_stats_path
        import pickle

        num_ensembles = 5
        scores = [NCSN(self.config).to(self.config.device) for _ in range(num_ensembles)]
        scores = [torch.nn.DataParallel(score) for score in scores]

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        fids = {}
        for ckpt in tqdm.tqdm(range(self.config.fast_fid.begin_ckpt, self.config.fast_fid.end_ckpt + 1, 5000),
                              desc="processing ckpt"):
            begin_ckpt = max(self.config.fast_fid.begin_ckpt, ckpt - (num_ensembles - 1) * 5000)
            index = 0
            for i in range(begin_ckpt, ckpt + 5000, 5000):
                states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{i}.pth'),
                                    map_location=self.config.device)
                scores[index].load_state_dict(states[0])
                scores[index].eval()
                index += 1

            def scorenet(x, labels):
                num_ckpts = (ckpt - begin_ckpt) // 5000 + 1
                return sum([scores[i](x, labels) for i in range(num_ckpts)]) / num_ckpts

            num_iters = self.config.fast_fid.num_samples // self.config.fast_fid.batch_size
            output_path = os.path.join(self.args.image_folder, 'ckpt_{}'.format(ckpt))
            os.makedirs(output_path, exist_ok=True)
            for i in range(num_iters):
                init_samples = torch.rand(self.config.fast_fid.batch_size, self.config.data.channels,
                                          self.config.data.image_size, self.config.data.image_size,
                                          device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, scorenet, sigmas,
                                                       self.config.fast_fid.n_steps_each,
                                                       self.config.fast_fid.step_lr,
                                                       verbose=self.config.fast_fid.verbose,
                                                       denoise=self.config.sampling.denoise)

                final_samples = all_samples[-1]
                for id, sample in enumerate(final_samples):
                    sample = sample.view(self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_size)

                    sample = inverse_data_transform(self.config, sample)

                    save_image(sample, os.path.join(output_path, 'sample_{}.png'.format(id)))

            stat_path = get_fid_stats_path(self.args, self.config, download=True)
            fid = get_fid(stat_path, output_path)
            fids[ckpt] = fid
            print("ckpt: {}, fid: {}".format(ckpt, fid))

        with open(os.path.join(self.args.image_folder, 'fids.pickle'), 'wb') as handle:
            pickle.dump(fids, handle, protocol=pickle.HIGHEST_PROTOCOL)
