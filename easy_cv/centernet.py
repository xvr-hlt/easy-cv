import numpy as np
from . import loss
import torch


class CenterNetLoss(torch.nn.Module):
    def __init__(self, hm_weight=1., wh_weight=.1, offset_weight=1.):
        super().__init__()
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.offset_weight = offset_weight
        self.heatmap_loss = loss.FocalLoss()
        self.wh_loss = loss.RegL1Loss()
        self.offset_loss = loss.RegL1Loss()

    def forward(self, output, batch):
        output['hm'] = torch.clamp(output['hm'].sigmoid(), min=1e-5, max=1-1e-5)
        hm_loss = self.heatmap_loss(output['hm'], batch['hm'])
        wh_loss = self.wh_loss(output['size'], batch['size'], batch['mask']) if self.wh_weight else 0.
        offset_loss = self.wh_loss(output['offset'], batch['offset'], batch['mask']) if self.offset_weight else 0.
        loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + self.offset_weight * offset_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss}
        if self.wh_weight:
            loss_stats['wh_loss'] = wh_loss
        if self.offset_weight:
            loss_stats['offset_loss'] = offset_loss
        return loss, loss_stats


class CenterNetMixin(object):
    def __init__(self, n_classes, downscale_ratio=4, min_gaussian_overlap=0.7):
        self.min_gaussian_overlap = min_gaussian_overlap
        self.downscale_ratio = downscale_ratio
        self.n_classes = n_classes
        self.loss = CenterNetLoss()

    def process(self, image, class_boxes):
        w, h, *_ = image.shape
        heatmap_w, heatmap_h = w // self.downscale_ratio, h // self.downscale_ratio
        heatmap = np.zeros((self.n_classes, heatmap_w, heatmap_h), dtype=np.float32)
        size = np.zeros((2, heatmap_w, heatmap_h), dtype=np.float32)
        offset = np.zeros((2, heatmap_w, heatmap_h), dtype=np.float32)
        mask = np.zeros((heatmap_w, heatmap_h), dtype=np.uint8)

        for class_ix, (x0, y0, x1, y1) in class_boxes:
            cx, cy, = (x0 + x1)/2, (y0 + y1)/2
            size_x, size_y = (x1 - x0)*heatmap_w, (y1 - y0)*heatmap_h
            heatmap_x = np.clip(int(cx*heatmap_w), 0, heatmap_w-1)
            heatmap_y = np.clip(int(cy*heatmap_h), 0, heatmap_h-1)
            self.draw_gaussian_inplace(
                heatmap[class_ix],
                (heatmap_x, heatmap_y),
                (size_y*self.downscale_ratio, size_x*self.downscale_ratio)
            )
            size[:, heatmap_x, heatmap_y] = (size_x, size_y)
            offset[:, heatmap_x, heatmap_y] = (cx*heatmap_w-heatmap_x), (cy*heatmap_h-heatmap_y)
            mask[heatmap_x, heatmap_y] = 1

        return {
            'image': image,
            'hm': heatmap,
            'mask': mask,
            'size': size,
            'offset': offset,
        }

    @staticmethod
    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_gaussian_inplace(self, heatmap, center, size, k=1):
        radius = max(0, int(self.gaussian_radius(size)))
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter/6)
        x, y = center
        width, height = heatmap.shape[0:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_heatmap = heatmap[x - left:x + right, y - top:y + bottom]
        masked_gaussian = gaussian[radius - left:radius + right, radius - top:radius + bottom]
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    def gaussian_radius(self, det_size):
        height, width = det_size
        min_overlap = self.min_gaussian_overlap

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)
        return min(r1, r2, r3)
