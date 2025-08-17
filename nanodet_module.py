import math

import torch
import numpy as np
import cv2

def _tensor_to_cv(img_t):
    """TorchCHW[0‑1]RGB→uint8HWCBGR (для cv2)."""
    img = img_t.detach().cpu().permute(1, 2, 0).numpy()           # -> HWC RGB
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def _iou_matrix(a, b):
    """IoU между двумя множествами боксов (N×4,M×4)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x11, y11, x12, y12 = np.split(a, 4, axis=1)
    x21, y21, x22, y22 = np.split(b, 4, axis=1)

    xa = np.maximum(x11, x21.T)
    ya = np.maximum(y11, y21.T)
    xb = np.minimum(x12, x22.T)
    yb = np.minimum(y12, y22.T)

    inter = np.clip(xb - xa, 0, None) * np.clip(yb - ya, 0, None)
    area_a = (x12 - x11) * (y12 - y11)
    area_b = (x22 - x21) * (y22 - y21)

    return inter / (area_a + area_b.T - inter + 1e-6)

def _preprocess(img, size, mean, scale):
    h, w = size
    blob = cv2.resize(img, (w, h)).astype(np.float32)
    return ((blob - mean) / scale).transpose(2,0,1)[None,...]

def softmax(x, axis=2):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def nms(boxes, scores, iou_thr=0.45):
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep=[]
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds+1]
    return keep

def make_grid_and_strides(in_h, in_w, strides):
    centers, stride_map = [], []
    for s in strides:
        fh = math.ceil(in_h/s); fw = math.ceil(in_w/s)
        yv, xv = np.meshgrid(np.arange(fh), np.arange(fw), indexing='ij')
        cx = (xv + 0.5)*s; cy = (yv + 0.5)*s
        pts = np.stack([cx, cy], -1).reshape(-1,2)
        centers.append(pts)
        stride_map.append(np.full((pts.shape[0],), s, dtype=np.float32))
    return np.concatenate(centers, 0), np.concatenate(stride_map, 0)


def postprocess(pred, orig_sz, in_sz, strides, conf_thr, num_classes):
    orig_h, orig_w = orig_sz
    in_h, in_w     = in_sz
    # 1) split
    cls_logits = pred[:, :num_classes]
    regs       = pred[:, num_classes:]          # (N,32)
    N, _       = cls_logits.shape

    # 2) grid & strides
    centers, stride_map = make_grid_and_strides(in_h, in_w, strides)

    # 3) class scores
    scores_all = 1 / (1 + np.exp(-cls_logits))
    class_ids  = np.argmax(scores_all, axis=1)
    scores     = scores_all[np.arange(N), class_ids]

    # 4) threshold
    mask       = scores > conf_thr
    scores     = scores[mask]
    class_ids  = class_ids[mask]
    regs       = regs[mask]
    centers    = centers[mask]
    stride_map = stride_map[mask]

    if scores.size == 0:
        return np.zeros((0,4)), np.array([]), np.array([])

    # 5) decode regs: reshape → (M,4,8)
    num_bins = 8
    regs     = regs.reshape(-1, 4, num_bins)
    probs    = softmax(regs, axis=2)               # (M,4,8)
    bins     = np.arange(num_bins, dtype=np.float32)
    dist     = (probs * bins).sum(axis=2) * stride_map[:,None]  # (M,4)
    l,t,r,b  = dist[:,0], dist[:,1], dist[:,2], dist[:,3]
    cx,cy    = centers[:,0], centers[:,1]
    x1,y1    = cx - l, cy - t
    x2,y2    = cx + r, cy + b
    boxes    = np.stack([x1,y1,x2,y2], axis=1)

    # 6) scale → original
    sx, sy = orig_w / in_w, orig_h / in_h
    boxes[:, [0,2]] *= sx
    boxes[:, [1,3]] *= sy

    # 7) NMS
    keep = nms(boxes, scores)
    return boxes[keep], scores[keep], class_ids[keep]

class BBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = cls
        self.conf = conf
        self.xyxy = xyxy


@torch.no_grad()
def nanodet_detect(model,
                   input_imgs,             # torch.Tensor B×3×H×W, 0‑1 RGB
                   cls_id_attacked: int,
                   clear_imgs=None,        # «чистые» изображения, такой же тензор
                   conf_thr: float = 0.25,
                   with_bbox: bool = False,
                   device: torch.device = torch.device('cpu')):
    """
    Возвращает:
        max_prob_obj_cls : Tensor[B]   – макс. score боксов атакуемого класса
        overlap_score    : Tensor[B]   – средний IoU с боксами на clean‑кадре
        bboxes           : list[list]  – боксы текущего кадра (для визуализации)
    """
    model.eval().to(device)

    batch = input_imgs.size(0)
    max_prob_obj_cls = []
    overlap_score    = []
    all_bboxes       = []

    # если есть clean‑кадры – считаем их вывод один раз
    if clear_imgs is not None:
        clean_boxes, _, clean_cls_ids = [], [], []
        for img_t in clear_imgs:
            orig = _tensor_to_cv(img_t)
            blob = _preprocess(orig, (model.input_size[1], model.input_size[2]),
                               mean=np.array([103.53, 116.28, 123.675], np.float32),
                               scale=np.array([57.375, 57.12, 58.395], np.float32))
            pred = model(torch.from_numpy(blob).to(device))[0].cpu().numpy()
            b, s, c = postprocess(pred, orig.shape[:2],
                                  model.input_size[1:], [8, 16, 32, 64],
                                  conf_thr, model.num_classes)
            clean_boxes.append(b); clean_cls_ids.append(c)

    for b_idx in range(batch):
        orig = _tensor_to_cv(input_imgs[b_idx])
        H, W  = model.input_size[1:]  # (H, W) из onnx‑модели
        blob  = _preprocess(orig, (H, W),
                            mean=np.array([103.53, 116.28, 123.675], np.float32),
                            scale=np.array([57.375, 57.12, 58.395], np.float32))
        pred  = model(torch.from_numpy(blob).to(device))[0].cpu().numpy()
        boxes, scores, cls_ids = postprocess(
            pred, orig.shape[:2], (H, W), [8, 16, 32, 64],
            conf_thr, model.num_classes
        )

        # --- максимальная вероятность для атакуемого класса
        mask_att = cls_ids == cls_id_attacked
        if mask_att.any():
            max_prob_obj_cls.append(torch.tensor(scores[mask_att].max(),
                                                 device=device,
                                                 dtype=torch.float32))
        else:
            max_prob_obj_cls.append(torch.tensor(0.0, device=device))

        # --- IoU‑overlap с clean‑кадром (если передан)
        if clear_imgs is not None:
            cb  = clean_boxes[b_idx]
            ci  = clean_cls_ids[b_idx]
            mask_clean = ci == cls_id_attacked
            if mask_clean.any() and mask_att.any():
                iou = _iou_matrix(boxes[mask_att], cb[mask_clean])
                overlap_score.append(torch.tensor(iou.max(), device=device,
                                                  dtype=torch.float32))
            else:
                overlap_score.append(torch.tensor(0.0, device=device))
        else:
            overlap_score.append(torch.tensor(0.0, device=device))

        # --- боксы для дальнейшей отрисовки
        if with_bbox:
            all_bboxes.append(list(zip(boxes, scores, cls_ids)))
        else:
            all_bboxes.append([])

    return (torch.stack(max_prob_obj_cls),
            torch.stack(overlap_score),
            all_bboxes)
