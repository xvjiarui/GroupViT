# Modified from the implementation of https://huggingface.co/akhaliq
import os
import sys
os.system("git clone https://github.com/NVlabs/GroupViT")
sys.path.insert(0, 'GroupViT')

import os.path as osp
from collections import namedtuple

import gradio as gr
import mmcv
import numpy as np
import torch
from datasets import build_text_transform
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from models import build_model
from omegaconf import read_write
from segmentation.datasets import (COCOObjectDataset, PascalContextDataset,
                                   PascalVOCDataset)
from segmentation.evaluation import (GROUP_PALETTE, build_seg_demo_pipeline,
                                     build_seg_inference)
from utils import get_config, load_checkpoint

os.chdir('GroupViT')
# checkpoint_url = 'https://github.com/xvjiarui/GroupViT/releases/download/v1.0.0/group_vit_gcc_yfcc_30e-74d335e6.pth'
checkpoint_url = 'https://github.com/xvjiarui/GroupViT/releases/download/v1.0.0/group_vit_gcc_yfcc_30e-879422e0.pth'
cfg_path = 'configs/group_vit_gcc_yfcc_30e.yml'
output_dir = 'demo/output'
device = 'cpu'
# vis_modes = ['first_group', 'final_group', 'input_pred_label']
vis_modes = ['input_pred_label', 'final_group']
output_labels = ['segmentation map', 'groups']
dataset_options = ['Pascal VOC', 'Pascal Context', 'COCO']
examples = [['Pascal VOC', '', 'demo/examples/voc.jpg'],
            ['Pascal Context', '', 'demo/examples/ctx.jpg'],
            ['COCO', 'rock', 'demo/examples/coco.jpg']]

PSEUDO_ARGS = namedtuple('PSEUDO_ARGS',
                         ['cfg', 'opts', 'resume', 'vis', 'local_rank'])

args = PSEUDO_ARGS(
    cfg=cfg_path, opts=[], resume=checkpoint_url, vis=vis_modes, local_rank=0)

cfg = get_config(args)

with read_write(cfg):
    cfg.evaluate.eval_only = True

model = build_model(cfg.model)
model = revert_sync_batchnorm(model)
model.to(device)
model.eval()

load_checkpoint(cfg, model, None, None)

text_transform = build_text_transform(False, cfg.data.text_aug, with_dc=False)
test_pipeline = build_seg_demo_pipeline()


def inference(dataset, additional_classes, input_img):
    if dataset == 'voc' or dataset == 'Pascal VOC':
        dataset_class = PascalVOCDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/pascal_voc12.py'
    elif dataset == 'coco' or dataset == 'COCO':
        dataset_class = COCOObjectDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/coco.py'
    elif dataset == 'context' or dataset == 'Pascal Context':
        dataset_class = PascalContextDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/pascal_context.py'
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    with read_write(cfg):
        cfg.evaluate.seg.cfg = seg_cfg
        cfg.evaluate.seg.opts = ['test_cfg.mode=whole']

    dataset_cfg = mmcv.Config()
    dataset_cfg.CLASSES = list(dataset_class.CLASSES)
    dataset_cfg.PALETTE = dataset_class.PALETTE.copy()

    if len(additional_classes) > 0:
        additional_classes = additional_classes.split(',')
        additional_classes = list(
            set(additional_classes) - set(dataset_cfg.CLASSES))
        dataset_cfg.CLASSES.extend(additional_classes)
        dataset_cfg.PALETTE.extend(GROUP_PALETTE[np.random.choice(
            list(range(len(GROUP_PALETTE))), len(additional_classes))])
    seg_model = build_seg_inference(model, dataset_cfg, text_transform,
                                    cfg.evaluate.seg)

    device = next(seg_model.parameters()).device
    # prepare data
    data = dict(img=input_img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(seg_model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    with torch.no_grad():
        result = seg_model(return_loss=False, rescale=True, **data)

    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)

    out_file_dict = dict()
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        for vis_mode in vis_modes:
            out_file = osp.join(output_dir, 'vis_imgs', vis_mode,
                                f'{vis_mode}.jpg')
            seg_model.show_result(img_show, img_tensor.to(device), result,
                                  out_file, vis_mode)
            out_file_dict[vis_mode] = out_file

    return [out_file_dict[mode] for mode in vis_modes]


title = 'GroupViT'

description = """
Gradio Demo for GroupViT: Semantic Segmentation Emerges from Text Supervision. \n
You may click on of the examples or upload your own image. \n
GroupViT could perform open vocabulary segmentation, you may input more classes,
e.g. "rock" is not in the COCO dataset, but you could input it for the giraffe image.
"""

article = """
<p style='text-align: center'>
<a href='https://arxiv.org/abs/2202.11094' target='_blank'>
GroupViT: Semantic Segmentation Emerges from Text Supervision
</a>
|
<a href='https://github.com/NVlabs/GroupViT' target='_blank'>Github Repo</a></p>
"""

gr.Interface(
    inference,
    inputs=[
        gr.inputs.Dropdown(dataset_options, type='value', label='Category list'),
        gr.inputs.Textbox(
            lines=1, placeholder=None, default='', label='More classes'),
        gr.inputs.Image(type='filepath')
    ],
    outputs=[gr.outputs.Image(label=label) for label in output_labels],
    title=title,
    description=description,
    article=article,
    examples=examples).launch(enable_queue=True)
