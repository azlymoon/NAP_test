import torch
import torch.nn as nn
import torch.nn.functional as F
from ensemble_tool.utils import denorm
from pytorchYOLOv4.tool.utils import load_class_names
from adversarialYolo.utils import do_detect as get_bbox_yolov2
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import time
import sys
import numpy as np

from GANLatentDiscovery.torch_tools.visualization import to_image
from GANLatentDiscovery.visualization import interpolate, interpolate_shift
from ipdb import set_trace as st

from pytorch_pretrained_detection import FasterrcnnResnet50

from nanodet_module import nanodet_detect


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)

def eval_rowPtach(generator, batch_size, device
                , latent_shift, alpha_latent
                , input_imgs, label, patch_scale, cls_id_attacked
                , denormalisation
                , model_name, detector
                , patch_transformer, patch_applier
                , by_rectangle
                , enable_rotation
                , enable_randomLocation
                , enable_crease
                , enable_projection
                , enable_rectOccluding
                , enable_blurred
                , enable_with_bbox
                , enable_show_plt
                , enable_clear_output
                , cls_conf_threshold
                , patch_mode = 0
                , multi_score = False
                , enable_no_random=False
                , fake_images_default=None):

                    
    

    fake_images = fake_images_default

    # enable_empty_patch == False: no patch
    enable_empty_patch =False
    if(patch_mode == 1):
        enable_empty_patch = True

    min_loss_det = 999
    for i_fimage in range(1):
        if not(enable_clear_output):
            adv_batch_t, adv_patch_set, msk_batch = patch_transformer(  adv_patch=fake_images[i_fimage], 
                                                                        lab_batch=label, 
                                                                        img_size=416,
                                                                        patch_mask=[],
                                                                        by_rectangle=by_rectangle,  
                                                                        do_rotate=enable_rotation, 
                                                                        rand_loc=enable_randomLocation, 
                                                                        with_black_trans=False, 
                                                                        scale_rate = patch_scale, 
                                                                        with_crease=enable_crease, 
                                                                        with_projection=enable_projection,
                                                                        with_rectOccluding=enable_rectOccluding,
                                                                        enable_empty_patch=enable_empty_patch,
                                                                        enable_no_random=enable_no_random,
                                                                        enable_blurred=enable_blurred)
            p_img_batch = patch_applier(input_imgs[0], adv_batch_t)
        else:
            p_img_batch = input_imgs

        # # test
        # print("label       size : "+str(label.size()))
        # print("label            : "+str(label))
        # print("fake_images size : "+str(fake_images.size()))
        # print("p_img_batch size : "+str(p_img_batch.size()))
        # # test
        # import numpy as np
        # def show(img):
        #     npimg = img.numpy()
        #     fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), 
        #                                     interpolation='nearest')
        #     fig.axes.get_xaxis().set_visible(False)
        #     fig.axes.get_yaxis().set_visible(False)
        # plt.ion()
        # plt.figure()
        # show(make_grid(fake_images.detach().cpu()[:16,:,:,:]))
        # plt.figure()
        # show(make_grid(p_img_batch.detach().cpu()[:16,:,:,:]))
        # plt.ioff()
        # plt.show()
        # sys.exit()

        # loss. TODO: посмотреть, нужно ли это вообще
        if(model_name == "yolov4"):
            max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
            # loss_det
            # default is multi_score. Here, "max_prob_cls" is "cls*obj"
            loss_det = torch.mean(max_prob_obj_cls)
            # loss_overlap
            loss_overlap = -torch.mean(overlap_score)
            
            # print(f"obtained loss_det: {loss_det}")
        elif(model_name == "yolov3"):
            max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
            # loss_det
            if multi_score:
                # multi = max_prob_obj * max_prob_cls
                loss_det = torch.mean(max_prob_obj_cls)
            else:
                loss_det = torch.mean(max_prob_obj_cls)
            # loss_overlap
            loss_overlap = -torch.mean(overlap_score)
        elif(model_name == "yolov2"):
            max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
            # loss_det
            if multi_score:
                # multi = max_prob_obj * max_prob_cls
                loss_det = torch.mean(max_prob_obj_cls)
            else:
                loss_det = torch.mean(max_prob_obj_cls)
            # loss_overlap
            loss_overlap = -torch.mean(overlap_score)
        elif(model_name == "fasterrcnn"):
            max_prob, bboxes = detector.detect(tensor_image_inputs=p_img_batch, cls_id_attacked=cls_id_attacked, threshold=0.5)
            # loss_det
            loss_det = torch.mean(max_prob)
            # no loss_overlap
            loss_overlap = torch.tensor(0.0).to(device)
        # elif model_name in ("yolov8", "yolov5"):
        elif ("yolov5" in model_name or "yolov8" in model_name or "yolov9" in model_name or "yolov10" in model_name):
            bboxes = detector(p_img_batch, verbose=False)
            combined_probs = []
            for box in bboxes[0].boxes:
                obj_prob = box.conf.cpu()
                if box.cls.cpu().item() == cls_id_attacked:
                        combined_probs.append(obj_prob)
            if combined_probs:
                    loss_det = torch.mean(torch.stack(combined_probs))
            else:
                loss_det = torch.tensor(0.0).to(device)

        elif (model_name == "nanodet"):
            max_prob_obj_cls, overlap_score, bboxes = nanodet_detect(
                model=detector,  # ← onnx2torch‑модель
                input_imgs=p_img_batch,
                cls_id_attacked=cls_id_attacked,
                clear_imgs=input_imgs,
                conf_thr=0, # TODO: чекнуть, возможно стоит изменить
                with_bbox=enable_with_bbox,
                device=device
            )
            loss_det = torch.mean(max_prob_obj_cls)
            loss_overlap = -torch.mean(overlap_score)

            # print(f"obtained loss_det: {loss_det}")
        else:
            raise Exception("Model not implemented")

        if(min_loss_det > loss_det):
            min_loss_det = loss_det
    
    loss_det = min_loss_det
    # print(f"final loss_det: {loss_det}")

    # draw bbox
    if enable_with_bbox and len(bboxes)>0:
        trans_2pilimage = transforms.ToPILImage()
        batch = p_img_batch.size()[0]
        for b in range(batch):
            img_pil = trans_2pilimage(p_img_batch[b].cpu())
            img_width = img_pil.size[0]
            img_height = img_pil.size[1]
            namesfile = 'pytorchYOLOv4/data/coco.names'
            class_names = load_class_names(namesfile)
            # sample first image
            # print("bbox : "+str(bbox))
            # if model_name in ("yolov8", "yolov5"):
            if ("yolov5" in model_name or "yolov8" in model_name or "yolov9" in model_name or "yolov10" in model_name):
                bbox = bboxes[b].boxes
            else:
                bbox = bboxes[b]
            for box in bbox:
                # print("box size : "+str(box.size()))
                # if model_name in ("yolov8", "yolov5"):
                if ("yolov5" in model_name or "yolov8" in model_name or "yolov9" in model_name or "yolov10" in model_name):
                    cls_id = box.cls.int()
                    cls_name = class_names[cls_id]
                    cls_conf = box.conf
                else:
                    cls_id = box[6].int()
                    cls_name = class_names[cls_id]
                    cls_conf = box[5]
                if(cls_id == cls_id_attacked):
                    if(cls_conf > cls_conf_threshold):
                        if(model_name == "yolov2"):
                            x_center    = box[0]
                            y_center    = box[1]
                            width       = box[2]
                            height      = box[3]
                            left        = (x_center.item() - width.item() / 2) * img_width
                            right       = (x_center.item() + width.item() / 2) * img_width
                            top         = (y_center.item() - height.item() / 2) * img_height
                            bottom      = (y_center.item() + height.item() / 2) * img_height
                        elif(model_name == "yolov4") or (model_name == "yolov3") or (model_name == "fasterrcnn"):
                            left        = int(box[0] * img_width)
                            right       = int(box[2] * img_width)
                            top         = int(box[1] * img_height)
                            bottom      = int(box[3] * img_height)
                        # elif model_name in ("yolov8", "yolov5"):
                        elif ("yolov5" in model_name or "yolov8" in model_name or "yolov9" in model_name or "yolov10" in model_name):
                            left, top, right, bottom = (
                                int(box.xyxy[0][0].cpu().item()),
                                int(box.xyxy[0][1].cpu().item()),
                                int(box.xyxy[0][2].cpu().item()),
                                int(box.xyxy[0][3].cpu().item()),
                            )
                        else:
                            raise Exception("Model not implemented: "+model_name)
                        # img with prediction
                        draw = ImageDraw.Draw(img_pil)
                        shape = [left, top, right, bottom]
                        draw.rectangle(shape, outline ="red")
                        # text
                        color = [255,0,0]
                        font = ImageFont.truetype("cmb10.ttf", int(min(img_width, img_height)/18))
                        sentence = str(cls_name)+" ("+str(round(float(cls_conf), 2))+")"
                        position = [left, top]
                        draw.text(tuple(position), sentence, tuple(color), font=font)
            if enable_show_plt:
                # show, debug
                plt.imshow(img_pil)
                plt.show()
            trans_2tensor = transforms.ToTensor()
            p_img_batch[b] = trans_2tensor(img_pil)
    
    return p_img_batch, fake_images, bboxes

def train_rowPtach(method_num, generator
                    , discriminator
                    , opt, batch_size, device
                    , latent_shift, alpha_latent
                    , input_imgs, label, patch_scale, cls_id_attacked
                    , denormalisation
                    , model_name, detector
                    , patch_transformer, patch_applier
                    , total_variation
                    , by_rectangle
                    , enable_rotation
                    , enable_randomLocation
                    , enable_crease
                    , enable_projection
                    , enable_rectOccluding
                    , enable_blurred
                    , enable_with_bbox
                    , enable_show_plt
                    , enable_clear_output
                    , weight_loss_tv
                    , weight_loss_overlap
                    , multi_score = False
                    , deformator=None
                    , fixed_latent_biggan=None
                    , max_value_latent_item=8
                    , enable_shift_deformator=True
                    ):
    # Clear generator gradients
    if method_num==2: #biggan
        opt.zero_grad()
        latent_shift.data = torch.clamp(latent_shift.data,-max_value_latent_item,max_value_latent_item)
        # Generate fake images
        # st()
        if not(deformator == None):
            # BigGAN
            rows = 1
            # set desired class for conditional GAN
            zs = torch.rand([rows, generator.dim_z] if type(generator.dim_z) == int else [rows] + generator.dim_z, device=device)
            # Do img = G(z + shift)
            z = zs[0]
            # print("z:",z[:10])
            if enable_shift_deformator:
                z = ((1-alpha_latent)*z) + (alpha_latent*fixed_latent_biggan)
                latent_shift = torch.clamp((max_value_latent_item*latent_shift),max=max_value_latent_item,min=-max_value_latent_item)
                interpolation_deformed = interpolate_shift(generator, z.unsqueeze(0),
                                        latent_shift=latent_shift.unsqueeze(0),
                                        deformator=deformator,
                                        device=device)
            else:
                # Directly, change z.  z = z + latent_shift
                z = ((1-alpha_latent)*z) + (alpha_latent*latent_shift)
                interpolation_deformed = interpolate_shift(generator, z.unsqueeze(0),
                                        latent_shift=torch.zeros_like(z.unsqueeze(0)).to(device),
                                        deformator=None,
                                        device=device)
            interpolation_deformed = interpolation_deformed.unsqueeze(0).to(device)   # torch.Size([1, 3, 128, 128])
            # print("dore",interpolation_deformed[0,0,:10,0])
            fake_images = (interpolation_deformed + 1) * 0.5
            # st()
        else:
            # Generate fake images
            latent = torch.rand(latent_shift.size(), device=device)
            latent_merged = ((1-alpha_latent)*latent) + (alpha_latent*latent_shift)
            if not(generator == None):
                fake_images = generator(latent_merged)
                if(denormalisation):
                    fake_images = denorm(fake_images) # torch.Size([batch, 3, 128, 128]) . batch=1
            else:
                fake_images = latent_merged.unsqueeze(0)

        min_loss_det = 999
        for i_fimage in range(1):
            if not(enable_clear_output):
                adv_batch_t, adv_patch_set, msk_batch = patch_transformer(  adv_patch=fake_images[i_fimage], 
                                                                            lab_batch=label, 
                                                                            img_size=416,
                                                                            patch_mask=[], 
                                                                            by_rectangle=by_rectangle, 
                                                                            do_rotate=enable_rotation, 
                                                                            rand_loc=enable_randomLocation, 
                                                                            with_black_trans=False, 
                                                                            scale_rate = patch_scale, 
                                                                            with_crease=enable_crease, 
                                                                            with_projection=enable_projection,
                                                                            with_rectOccluding=enable_rectOccluding,
                                                                            enable_blurred=enable_blurred)
                p_img_batch = patch_applier(input_imgs, adv_batch_t) # torch.Size([8, 14, 3, 416, 416])
                # st()

                # label       size : torch.Size([16, 14, 5])
                # input_imgs  size : torch.Size([16, 3, 416, 416])
                # adv_batch_t size : torch.Size([16, 14, 3, 416, 416])
                # p_img_batch size : torch.Size([16, 3, 416, 416])
                # adv_patch_set size :torch.Size([3, 300, 300])
                # print("p_img_batch:",p_img_batch[0,0,:10,0])
            else:
                p_img_batch = input_imgs
            if discriminator is not None:
                D_loss = discriminator(fake_images, torch.tensor([259], device=device).long())
            else:
                D_loss = 0
            # st()
            # loss.
            if(model_name == "yolov4"):
                max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
                # st()
                # np.save('gg', p_img_batch.cpu().detach().numpy())   
                                            #  gg=np.load('gg.npy')    gg2=p_img_batch.cpu().detach().numpy()  np.argwhere(gg!=gg2)
                # loss_det
                # default is multi_score. Here, "max_prob_cls" is "cls*obj"
                # print(max_prob_obj_cls)
                # st()
                loss_det = torch.mean(max_prob_obj_cls)
                # loss_overlap
                loss_overlap = -torch.mean(overlap_score)
                
                # print(f"obtained loss_det: {loss_det}")
            elif(model_name == "yolov3"):
                max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
                # loss_det
                if multi_score:
                    # multi = max_prob_obj * max_prob_cls
                    loss_det = torch.mean(max_prob_obj_cls)
                else:
                    loss_det = torch.mean(max_prob_obj_cls)
                # loss_overlap
                loss_overlap = -torch.mean(overlap_score)
            elif(model_name == "yolov2"):
                max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
                # loss_det
                if multi_score:
                    loss_det = torch.mean(max_prob_obj_cls)
                else:
                    loss_det = torch.mean(max_prob_obj_cls)
                # loss_overlap
                loss_overlap = -torch.mean(overlap_score)
            elif(model_name == "fasterrcnn"):
                max_prob, bboxes = detector.detect(tensor_image_inputs=p_img_batch, cls_id_attacked=cls_id_attacked, threshold=0.5)
                # loss_det
                loss_det = torch.mean(max_prob)
                # no loss_overlap
                loss_overlap = torch.tensor(0.0).to(device)
            # elif model_name in ("yolov8", "yolov5"):
            # elif "yolov" in model_name:
            elif ("yolov5" in model_name or "yolov8" in model_name or "yolov9" in model_name or "yolov10" in model_name):
                bboxes = detector(p_img_batch, verbose=False)
                combined_probs = []
                for box in bboxes[0].boxes:
                    obj_prob = box.conf.cpu()
                    if box.cls.cpu().item() == cls_id_attacked:
                            combined_probs.append(obj_prob)
                if combined_probs:
                        loss_det = torch.mean(torch.stack(combined_probs)).to(device)
                else:
                    loss_det = torch.tensor(0.0).to(device)
                # no loss_overlap # TODO: Check what is this (?)
                loss_overlap = torch.tensor(0.0).to(device)
                
                # print(f"obtained loss_det: {loss_det}")
            elif (model_name == "nanodet"):
                max_prob_obj_cls, overlap_score, bboxes = nanodet_detect(
                    model=detector,  # ← onnx2torch‑модель
                    input_imgs=p_img_batch,
                    cls_id_attacked=cls_id_attacked,
                    clear_imgs=input_imgs,
                    conf_thr=0.2, # TODO: чекнуть, возможно стоит изменить
                    with_bbox=enable_with_bbox,
                    device=device
                )
                loss_det = torch.mean(max_prob_obj_cls)
                loss_overlap = -torch.mean(overlap_score)
            else:
                raise Exception("Model not implemented")

            if(min_loss_det > loss_det):
                min_loss_det = loss_det

        # loss: total variation
        loss_tv = total_variation(fake_images[0])
        # st()
        # loss
        if discriminator is not None:
            loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv) + 1e-3 * D_loss
        else:
            loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv) 
            # print(f"[DEBUG] Final loss before backward: {loss}")

        # Update generator weights
        loss.backward()
        opt.step()

        # draw bbox at output
        if enable_with_bbox and len(bboxes)>0:
            trans_2pilimage = transforms.ToPILImage()
            batch = p_img_batch.size()[0]
            for b in range(batch):
                img_pil = trans_2pilimage(p_img_batch[b].cpu())
                img_width = img_pil.size[0]
                img_height = img_pil.size[1]
                namesfile = 'pytorchYOLOv4/data/coco.names'
                class_names = load_class_names(namesfile)
                # sample first image
                bbox = bboxes[b]
                # print("bbox : "+str(bbox))
                for box in bbox:
                    # print("box size : "+str(box.size()))
                    if model_name == 'nanodet':
                        cls_id = box[2]
                        cls_conf = box[1]
                    else:
                        cls_id = box.boxes.cls.int()
                        cls_conf = box.boxes.conf
                    cls_name = class_names[cls_id]
                    if(cls_id == cls_id_attacked):
                        if(model_name == "yolov2"):
                            x_center    = box[0]
                            y_center    = box[1]
                            width       = box[2]
                            height      = box[3]
                            left        = (x_center.item() - width.item() / 2) * img_width
                            right       = (x_center.item() + width.item() / 2) * img_width
                            top         = (y_center.item() - height.item() / 2) * img_height
                            bottom      = (y_center.item() + height.item() / 2) * img_height
                        elif(model_name == "yolov4") or (model_name == "yolov3" or (model_name == "fasterrcnn")):
                            left        = int(box[0] * img_width)
                            right       = int(box[2] * img_width)
                            top         = int(box[1] * img_height)
                            bottom      = int(box[3] * img_height)
                        # elif model_name in ("yolov8", "yolov5"):
                        elif ("yolov5" in model_name or "yolov8" in model_name or "yolov9" in model_name or "yolov10" in model_name):
                            left, top, right, bottom = (
                                int(box.boxes.xyxy[0][0].cpu().item()),
                                int(box.boxes.xyxy[0][1].cpu().item()),
                                int(box.boxes.xyxy[0][2].cpu().item()),
                                int(box.boxes.xyxy[0][3].cpu().item()),
                            )
                        elif (model_name == 'nanodet'): # TODO: если с границами че-то не так, то менять стоит здесь
                            left, top, right, bottom = (
                                int(box[0][0]),
                                int(box[0][1]),
                                int(box[0][2]),
                                int(box[0][3]),
                            )
                        else:
                            raise Exception("Model not implemented")
                        # img with prediction
                        draw = ImageDraw.Draw(img_pil)
                        shape = [left, top, right, bottom]
                        draw.rectangle(shape, outline ="red")
                        # text
                        color = [255,0,0]
                        font = ImageFont.truetype("cmb10.ttf", int(min(img_width, img_height)/18))
                        sentence = str(cls_name)+" ("+str(round(float(cls_conf), 2))+")"
                        position = [left, top]
                        draw.text(tuple(position), sentence, tuple(color), font=font)
                if enable_show_plt:
                    # show, debug
                    plt.imshow(img_pil)
                    plt.show()
                trans_2tensor = transforms.ToTensor()
                p_img_batch[b] = trans_2tensor(img_pil)
        # print("loss_det:",loss)
        # st()
        return loss_det, loss_overlap, loss_tv, p_img_batch, fake_images, D_loss
    
    #######################################################################################
    ## From here, the code is for StyleGAN2 (Which does not work at the moment) (it should now) (it does)
    elif method_num==3: # stylegan2
        opt.zero_grad()
        # latent_shift = torch.clamp(latent_shift,-4,4)
        # Generate fake images
        rows = 1
        # set desired class for conditional GAN
        zs = torch.rand([rows, generator.latent_size] if type(generator.latent_size) == int else [rows] + generator.latent_size, device=device)
        # Do img = G(z + shift)
        # Directly, change z.  z = z + latent_shift
        z = latent_shift
        interpolation_deformed = generator(z.unsqueeze(0))
        # st()
        interpolation_deformed = interpolation_deformed.to(device)   # torch.Size([1, 3, 128, 128])
        fake_images = (interpolation_deformed+1) * 0.5
        fake_images = torch.clamp(fake_images,0,1)
        # st()
        # images = utils.tensor_to_PIL(
        #     generated, pixel_min=args.pixel_min, pixel_max=args.pixel_max)

        min_loss_det = 999
        for i_fimage in range(1):
            if not(enable_clear_output):
                adv_batch_t, adv_patch_set, msk_batch = patch_transformer(  adv_patch=fake_images[i_fimage], 
                                                                            lab_batch=label, 
                                                                            img_size=416,
                                                                            patch_mask=[], 
                                                                            by_rectangle=by_rectangle, 
                                                                            do_rotate=enable_rotation, 
                                                                            rand_loc=enable_randomLocation, 
                                                                            with_black_trans=False, 
                                                                            scale_rate = patch_scale, 
                                                                            with_crease=enable_crease, 
                                                                            with_projection=enable_projection,
                                                                            with_rectOccluding=enable_rectOccluding,
                                                                            enable_blurred=enable_blurred)
                p_img_batch = patch_applier(input_imgs, adv_batch_t)
                # label       size : torch.Size([16, 14, 5])
                # input_imgs  size : torch.Size([16, 3, 416, 416])
                # adv_batch_t size : torch.Size([16, 14, 3, 416, 416])
                # p_img_batch size : torch.Size([16, 3, 416, 416])
                # adv_patch_set size :torch.Size([3, 300, 300])
            else:
                p_img_batch = input_imgs
            if discriminator is not None:
                D_loss = discriminator(fake_images, torch.tensor([259], device=device).long())
            else:
                D_loss = 0
            # st()
            # loss.
            if(model_name == "yolov4"):
                max_prob_obj_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
                # loss_det
                # default is multi_score. Here, "max_prob_cls" is "cls*obj"
                loss_det = torch.mean(max_prob_obj_cls)
                # loss_overlap
                loss_overlap = -torch.mean(overlap_score)
            elif(model_name == "yolov3"):
                max_prob_obj, max_prob_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
                # loss_det
                if multi_score:
                    multi = max_prob_obj * max_prob_cls
                    loss_det = torch.mean(multi)
                else:
                    loss_det = torch.mean(max_prob_obj)
                # loss_overlap
                loss_overlap = -torch.mean(overlap_score)
            elif(model_name == "yolov2"):
                max_prob_obj, max_prob_cls, overlap_score, bboxes = detector.detect(input_imgs=p_img_batch, cls_id_attacked=cls_id_attacked, clear_imgs=input_imgs, with_bbox=enable_with_bbox)
                # loss_det
                if multi_score:
                    multi = max_prob_obj * max_prob_cls
                    loss_det = torch.mean(multi)
                else:
                    loss_det = torch.mean(max_prob_obj)
                # loss_overlap
                loss_overlap = -torch.mean(overlap_score)
            # elif model_name in ("yolov8", "yolov5"):
            elif ("yolov5" in model_name or "yolov8" in model_name):
                bboxes = detector(p_img_batch, verbose=False)
                combined_probs = []
                for box in bboxes[0].boxes:
                    obj_prob = box.conf.cpu()
                    if box.cls.cpu().item() == cls_id_attacked:
                            combined_probs.append(obj_prob)
                if combined_probs:
                        loss_det = torch.mean(torch.stack(combined_probs)).to(device)
                else:
                    loss_det = torch.tensor(0.0).to(device)
                # no loss_overlap # TODO: Check what is this (?)
                loss_overlap = torch.tensor(0.0).to(device)
            else:
                raise Exception(f"Model not implemented: {model_name}")
            if(model_name == "fasterrcnn"):
                max_prob, max_prob, bboxes = FasterrcnnResnet50(tensor_image_inputs=p_img_batch, device=device, cls_id_attacked=cls_id_attacked, threshold=0.5)
                # loss_det
                loss_det = torch.mean(max_prob)
                # no loss_overlap
                loss_overlap = torch.tensor(0).to(device)

            if(min_loss_det > loss_det):
                min_loss_det = loss_det

        # loss: total variation
        loss_tv = total_variation(fake_images[0])
        # st()
        # loss
        if discriminator is not None:
            loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv) + 1e-3 * D_loss
        else:
            loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv) 

        # Update generator weights
        loss.backward()
        opt.step()

        # draw bbox at output
        if enable_with_bbox and len(bboxes)>0:
            trans_2pilimage = transforms.ToPILImage()
            batch = p_img_batch.size()[0]
            for b in range(batch):
                img_pil = trans_2pilimage(p_img_batch[b].cpu())
                img_width = img_pil.size[0]
                img_height = img_pil.size[1]
                namesfile = 'pytorchYOLOv4/data/coco.names'
                class_names = load_class_names(namesfile)
                # sample first image
                bbox = bboxes[b]
                # print("bbox : "+str(bbox))
                for box in bbox:
                    # print("box size : "+str(box.size()))
                    cls_id = box[6].int()
                    cls_name = class_names[cls_id]
                    cls_conf = box[5]
                    if(cls_id == cls_id_attacked):
                        if(model_name == "yolov2"):
                            x_center    = box[0]
                            y_center    = box[1]
                            width       = box[2]
                            height      = box[3]
                            left        = (x_center.item() - width.item() / 2) * img_width
                            right       = (x_center.item() + width.item() / 2) * img_width
                            top         = (y_center.item() - height.item() / 2) * img_height
                            bottom      = (y_center.item() + height.item() / 2) * img_height
                        if(model_name == "yolov4") or (model_name == "yolov3" or (model_name == "fasterrcnn")):
                            left        = int(box[0] * img_width)
                            right       = int(box[2] * img_width)
                            top         = int(box[1] * img_height)
                            bottom      = int(box[3] * img_height)
                        # img with prediction
                        draw = ImageDraw.Draw(img_pil)
                        shape = [left, top, right, bottom]
                        draw.rectangle(shape, outline ="red")
                        # text
                        color = [255,0,0]
                        font = ImageFont.truetype("cmb10.ttf", int(min(img_width, img_height)/18))
                        sentence = str(cls_name)+" ("+str(round(float(cls_conf), 2))+")"
                        position = [left, top]
                        draw.text(tuple(position), sentence, tuple(color), font=font)
                if enable_show_plt:
                    # show, debug
                    plt.imshow(img_pil)
                    plt.show()
                trans_2tensor = transforms.ToTensor()
                p_img_batch[b] = trans_2tensor(img_pil)
        # print("loss:",loss_det)
        return loss_det, loss_overlap, loss_tv, p_img_batch, fake_images, D_loss



# # test
# import numpy as np
# def show(img):
#     npimg = img.numpy()
#     fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), 
#                                     interpolation='nearest')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
# plt.ion()
# plt.figure()
# show(make_grid(p_img_batch.detach().cpu()[:16,:,:,:]))
# plt.ioff()
# plt.show()
# sys.exit()
