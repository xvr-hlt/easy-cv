import numpy as np


class CenterNetMixin(object):
    def __init__(self, n_classes, downscale_ratio=4, min_gaussian_overlap=0.7):
        self.min_gaussian_overlap = min_gaussian_overlap
        self.downscale_ratio = downscale_ratio
        self.n_classes = n_classes

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
