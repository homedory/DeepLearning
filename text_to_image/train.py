import torch
import matplotlib.pyplot as plt
import torchvision
import os
import numpy as np
import clip
import torchvision.transforms as T
import time
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import BCELoss
from read_dataset import ZipDataset
from dataloader import get_dataloader
from network import Generator, Discriminator, weight_init
from train_utils import *

d_losses = []
g_losses = []

cos_sim = torch.nn.CosineSimilarity(dim=0)


def contrastive_loss_G(fake_image, clip_model, txt_embedding, device, tau=0.5):

    denorm_fake_image = denormalize_image(fake_image)
    reshaped_img = clip_transform(224)(denorm_fake_image)
    reshaped_img = custom_reshape(reshaped_img)
    normed_img = clip_preprocess()(reshaped_img).to(device)

    img_embedding = clip_model.encode_image(normed_img).float()
    txt_embedding = txt_embedding.float()

    sim = torch.cosine_similarity(txt_embedding.unsqueeze(1), img_embedding.unsqueeze(0), dim=-1) / tau
    sim = torch.diagonal(F.softmax(sim, 1))
    L_cont = -tau * torch.log(sim)
    L_cont = torch.mean(L_cont)

    return L_cont


def contrastive_loss_D(g_out_align, txt_embedding, tau=0.5):

    model_features = normalize(g_out_align) 

    sim = torch.cosine_similarity(txt_embedding.unsqueeze(1), model_features.unsqueeze(0), dim=-1) / tau
    sim = torch.diagonal(F.softmax(sim, 1))
    L_cont = -tau * torch.log(sim)
    L_cont = torch.mean(L_cont)

    return L_cont


def D_loss(real_image, fake_image, model_D, loss_fn, 
               use_uncond_loss, use_contrastive_loss, 
               gamma,
               mu, txt_feature,
               d_fake_label, d_real_label):
    
    loss_d_comp = {}



    if use_uncond_loss:
        out_fake_uncond, _ = model_D(fake_image.detach()) # detach here?
        out_real_uncond, _ = model_D(real_image)

        loss_d_comp['d_loss_fake_uncond'] = loss_fn(out_fake_uncond, d_fake_label)
        loss_d_comp['d_loss_real_uncond'] = loss_fn(out_real_uncond, d_real_label)

    out_fake_cond, d_out_align_fake = model_D(fake_image.detach(), mu)
    out_real_cond, d_out_align_real = model_D(real_image, mu)

    loss_d_comp['d_loss_fake_cond'] = loss_fn(out_fake_cond, d_fake_label)
    loss_d_comp['d_loss_real_cond'] = loss_fn(out_real_cond, d_real_label)
    
    if use_contrastive_loss:
        loss_d_comp['d_loss_fake_cond_contrastive'] = gamma * contrastive_loss_D(d_out_align_fake, txt_feature)
        loss_d_comp['d_loss_real_cond_contrastive'] = gamma * contrastive_loss_D(d_out_align_real, txt_feature)

    d_loss = gather_all(loss_d_comp)
    return d_loss


def G_loss(fake_image, model_D, loss_fn,
           use_uncond_loss, use_contrastive_loss,
           clip_model, gamma, lam, device,
           mu, txt_feature,
           g_label):
    
    loss_g_comp = {}


    if use_uncond_loss:
        out_uncond, align_out_uncond = model_D(fake_image)
        loss_g_comp['g_loss_uncond'] = loss_fn(out_uncond, g_label)

    out_cond, g_out_align = model_D(fake_image, mu)
    loss_g_comp['g_loss_cond'] = loss_fn(out_cond, g_label)

    if use_contrastive_loss:
        if fake_image.shape[-1] >= 256: 
            loss_g_comp['g_loss_cond_contrastive'] = lam * contrastive_loss_G(fake_image, clip_model, txt_feature, device)
        loss_g_comp['d_loss_cond_contrastive'] = gamma * contrastive_loss_D(g_out_align, txt_feature)


    g_loss = gather_all(loss_g_comp)
    return g_loss


def train_step(train_loader, noise_dim, device, model_G, model_D_lst, optim_g, optim_d_lst, 
               loss_fn, num_stage, use_uncond_loss, use_contrastive_loss, report_interval, 
               clip_model, gamma, lam):
    
    d_loss_train = 0
    g_loss_train = 0

    for iter, batch in enumerate(train_loader):
        real_imgs, img_feature, txt_feature = batch

        if iter == 0: save_txt_feature = txt_feature

        BATCH_SIZE = real_imgs[-1].shape[0]
        for i in range(num_stage): real_imgs[i] = real_imgs[i].to(device)

        img_feature = img_feature.to(device)
        txt_feature = txt_feature.to(device)

        level_perturb = 0.1
        noise = torch.randn_like(img_feature)
        pseudo_text_feature = img_feature + level_perturb * noise * torch.norm(img_feature, dim=1, keepdim=True) / torch.norm(noise, dim=1, keepdim=True)

        d_fake_label = torch.full([img_feature.shape[0], 1], 0.0, dtype=img_feature.dtype, device=device)
        d_real_label = torch.full([img_feature.shape[0], 1], 1.0, dtype=img_feature.dtype, device=device)
        g_label = d_real_label

        # Optmize Discriminator
        noise = torch.randn(BATCH_SIZE, noise_dim).to(device)
        fake_images, mu, log_sigma = model_G(txt_feature, noise)
        d_loss = 0

        for i in range(num_stage):
            optim_d = optim_d_lst[i]
            optim_d.zero_grad()

            d_loss_i = D_loss(real_imgs[i], fake_images[i], model_D_lst[i], loss_fn, 
                              use_uncond_loss, use_contrastive_loss,
                              gamma,
                              mu, txt_feature,
                              d_fake_label, d_real_label)

            d_loss += d_loss_i.detach().item()
            d_loss_i.backward(retain_graph=True)
            optim_d.step()
            d_loss_train += d_loss_i.item()

        # Optimize Generator
        optim_g.zero_grad()
        noise = torch.randn(BATCH_SIZE, noise_dim).to(device)
        fake_images, mu, log_sigma = model_G(txt_feature, noise)
        g_loss = 0

        for i in range(num_stage):
            g_loss_i = G_loss(fake_images[i], model_D_lst[i], loss_fn,
                              use_uncond_loss, use_contrastive_loss,
                              clip_model, gamma, lam, device,
                              mu, txt_feature,
                              g_label)
            g_loss += g_loss_i

        # Calculation of L_CA. Do NOT modify.
        aug_loss = KL_divergence(mu, log_sigma)
        g_loss += (1.0) * aug_loss
        g_loss.backward()
        optim_g.step()
        g_loss_train += g_loss.item()

        # Report
        if iter % report_interval == 0 and iter >= report_interval:
            print(f"    Iteration {iter} \t d_loss: {(d_loss):.4f}, g_loss: {(g_loss.item()):.4f}")
    
    d_loss_train /= len(train_loader)
    g_loss_train /= len(train_loader)
    d_losses.append(d_loss_train)
    g_losses.append(g_loss_train)

    return d_loss_train, g_loss_train, save_txt_feature


def train(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    clip_embedding_dim = args.clip_embedding_dim # Dimension of c_txt, default: 512 (CLIP ViT-B/32)
    projection_dim = args.projection_dim # Dimension of \hat{c_txt} extracted from CANet, default: 128
    noise_dim = args.noise_dim # Dimension of noise z ~ N(0, 1), default: 100 
    g_in_chans = 1024 # Equal to Ng
    g_out_chans = 3 # Fixed
    d_in_chans = 64 # Equal to Nd
    d_out_chans = 1 # Fixed
    num_stage = args.num_stage # default: 3
    use_uncond_loss = args.use_uncond_loss
    use_contrastive_loss = args.use_contrastive_loss
    report_interval = args.report_interval # default: 100

    save_hyp(args, g_in_chans, g_out_chans, d_in_chans, d_out_chans)

    print("Loading dataset")
    train_dataset = ZipDataset(args.train_data, num_stage)
    train_loader = get_dataloader(args=args, dataset=train_dataset, is_train=True)
    print("finish")

    G = Generator(clip_embedding_dim, projection_dim, noise_dim, g_in_chans, g_out_chans, num_stage, device).to(device)
    G.apply(weight_init)

    D_lst = [
        Discriminator(projection_dim, g_out_chans, d_in_chans, d_out_chans, clip_embedding_dim, curr_stage, device).to(device)
        for curr_stage in range(num_stage)
    ]
    for D in D_lst:
        D.apply(weight_init)
   
    if args.resume_checkpoint_path is not None and args.resume_epoch != -1:
        load_checkpoint(G, D_lst, args.resume_checkpoint_path, args.resume_epoch)
        print('Resumed from saved checkpoint')

    lr = args.learning_rate
    num_epochs = args.num_epochs

    # NOTE: You may try different optimizer setting or use learning rate schduler
    optim_g = Adam(G.parameters(), lr = lr, betas = (0.5, 0.999))
    optim_d_lst = [
        Adam(D_lst[curr_stage].parameters(), lr = lr, betas = (0.5, 0.999))
        for curr_stage in range(num_stage)
    ]
    loss_fn = BCELoss()

    clip_model, _ = clip.load("ViT-B/32", device=device)

    for epoch in range(args.resume_epoch + 1, num_epochs):
        print(f"Epoch: {epoch} start")
        start_time = time.time()
        d_loss, g_loss, txt_feature = train_step(train_loader, noise_dim, device, G, D_lst, optim_g, optim_d_lst, 
                                                 loss_fn, num_stage, use_uncond_loss, use_contrastive_loss, report_interval,
                                                 clip_model, gamma=5, lam=10)
        end_time = time.time()
        print(f"Epoch: {epoch} \t d_loss: {d_loss:.4f} \t g_loss: {g_loss:.4f} \t esti. time: {(end_time - start_time):.2f}s")

        # sampling : generate fake images and save
        with torch.no_grad():
            z = torch.randn(txt_feature.shape[0], noise_dim).to(device)
            txt_feature = txt_feature.to(device)

            fake_images, _, _ = G(txt_feature, z)
            fake_image = fake_images[-1].detach().cpu() # visulize only the high-res images
            epoch_ret = torchvision.utils.make_grid(fake_image, padding=2, normalize=True)
            torchvision.utils.save_image(epoch_ret, os.path.join(args.result_path, f"epoch_{epoch}.png"))

        # save checkpoint
        save_model(args, G, D_lst, epoch, num_stage)
