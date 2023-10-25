import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
import fractions
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import torchvision.utils as vutils


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
if opt.fp16:
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    if len(opt.gpu_ids) > 0:
        optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
        if not opt.no_domain_loss:
            optimizer_D_Domain = model.module.optimizer_D_Domain
    else:
        optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D
        if not opt.no_domain_loss:
            optimizer_D_Domain = model.optimizer_D_Domain

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

print("display_delta", display_delta)
print("print_delta", print_delta)
print("save_delta", save_delta)

torch.cuda.empty_cache()
total_time_start = time.time()
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        image_contrastive_list = []
        for i in range(len(data['image_contrastive'])):
            image_contrastive_list.append(Variable(data['image_contrastive'][i]))

        a = time.time()
        losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                  Variable(data['image']), Variable(data['feat']),
                                  Variable(data['reference']), image_contrastive_list, infer=save_fake)

        # sum per device losses
        losses = [torch.mean(x).unsqueeze(0) if not isinstance(x, int) else x for x in losses]
        if len(opt.gpu_ids) > 0:
            loss_dict = dict(zip(model.module.loss_names, losses))
        else:
            loss_dict = dict(zip(model.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0) + loss_dict.get('G_VGG_Refine', 0) + \
                     loss_dict.get('G_Ctx', 0) + loss_dict.get('G_Style', 0) + loss_dict.get('G_Style_Contrastive', 0) + loss_dict.get('G_Perc', 0) + loss_dict.get('G_Contrastive', 0) + \
                     loss_dict['G_Rec_ske'] + loss_dict.get('G_GAN_Domain_ref', 0) + loss_dict.get('G_GAN_Domain_ske', 0)

        if not opt.no_domain_loss:
            loss_D_Domain = (loss_dict['D_GAN_Domain_ref'] + loss_dict['D_GAN_Domain_ske']) * 0.5

        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_G.backward()
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_D.backward()
        optimizer_D.step()

        # update domain discriminator weights
        if not opt.no_domain_loss:
            optimizer_D_Domain.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_D_Domain, optimizer_D_Domain) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_D_Domain.backward()
            optimizer_D_Domain.step()

        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated[0].data[0])),
                                   ('recons_image', util.tensor2im(generated[1].data[0])),
                                   ('real_image', util.tensor2im(data['image'][0])),
                                   ('reference_image', util.tensor2im(data['reference'][0]))
                                   ])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            if len(opt.gpu_ids) > 0:
                model.module.save('latest')
            else:
                model.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
        if epoch_iter >= dataset_size:
            break

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %.4f sec\n' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        if len(opt.gpu_ids) > 0:
            model.module.save('latest')
            model.module.save(epoch)
        else:
            model.save('latest')
            model.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        if len(opt.gpu_ids) > 0:
            model.module.update_fixed_params()
        else:
            model.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        if len(opt.gpu_ids) > 0:
            model.module.update_learning_rate()
        else:
            model.update_learning_rate()

print('End of training [%s], Time Taken: %.4f sec' % (opt.name, time.time() - total_time_start))
print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
if len(opt.gpu_ids) > 0:
    model.module.save('latest')
else:
    model.save('latest')
vutils.save_image(data['image'], '%s/%s/%d_image.png' % (opt.checkpoints_dir, opt.name, epoch_iter),normalize=True)
vutils.save_image(generated[0].data, '%s/%s/%d_fake.png' % (opt.checkpoints_dir, opt.name, epoch_iter),normalize=True)
vutils.save_image(data['label'], '%s/%s/%d_label.png' % (opt.checkpoints_dir, opt.name, epoch_iter),normalize=True)
np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
print('End of training [%s], Time Taken: %.4f sec' % (opt.name, time.time() - total_time_start))
