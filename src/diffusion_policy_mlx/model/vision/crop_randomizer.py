"""
CropRandomizer — random crop augmentation for vision inputs.

In training mode: applies random spatial crops.
In eval mode: applies deterministic center crops.

Input/output follow NCHW convention (B, C, H, W).
"""

import mlx.core as mx
import mlx.nn as nn


class CropRandomizer(nn.Module):
    """Randomly sample crops during training, center crop during eval.

    Args:
        input_shape: Shape of input *without* batch dimension, i.e. ``(C, H, W)``.
        crop_height: Height of the crop.
        crop_width: Width of the crop.
        num_crops: Number of random crops per image (default 1).
        pos_enc: If True, append positional encoding channels (not implemented yet).
    """

    def __init__(
        self,
        input_shape,
        crop_height: int,
        crop_width: int,
        num_crops: int = 1,
        pos_enc: bool = False,
    ):
        super().__init__()
        assert len(input_shape) == 3, f"Expected (C, H, W), got shape with {len(input_shape)} dims"
        assert crop_height < input_shape[1], "crop_height must be < input height"
        assert crop_width < input_shape[2], "crop_width must be < input width"

        self.input_shape = tuple(input_shape)
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc
        # training flag — toggled by .train() / .eval()
        self._training = True

    def train(self):
        self._training = True
        return self

    def eval(self):
        self._training = False
        return self

    @property
    def training(self):
        return self._training

    def __call__(self, x):
        """
        Args:
            x: (B, C, H, W) in NCHW convention.

        Returns:
            Cropped tensor of shape (B * num_crops, C, crop_h, crop_w).
            When num_crops == 1, shape is (B, C, crop_h, crop_w).
        """
        if self._training:
            return self._random_crop(x)
        else:
            return self._center_crop(x)

    def _random_crop(self, x):
        """Random spatial crop for each image in the batch."""
        B, C, H, W = x.shape
        ch, cw = self.crop_height, self.crop_width

        crops = []
        for _ in range(self.num_crops):
            # Sample random top-left corners for each image in batch
            h_starts = mx.random.randint(0, H - ch, shape=(B,))
            w_starts = mx.random.randint(0, W - cw, shape=(B,))

            # Extract crops per image
            batch_crops = []
            for b in range(B):
                h0 = h_starts[b].item()
                w0 = w_starts[b].item()
                crop = x[b, :, h0 : h0 + ch, w0 : w0 + cw]
                batch_crops.append(crop)
            # (B, C, ch, cw)
            crops.append(mx.stack(batch_crops, axis=0))

        if self.num_crops == 1:
            return crops[0]

        # (num_crops, B, C, ch, cw) → (B * num_crops, C, ch, cw)
        stacked = mx.stack(crops, axis=1)  # (B, num_crops, C, ch, cw)
        return stacked.reshape(B * self.num_crops, C, ch, cw)

    def _center_crop(self, x):
        """Deterministic center crop."""
        B, C, H, W = x.shape
        h_off = (H - self.crop_height) // 2
        w_off = (W - self.crop_width) // 2
        out = x[:, :, h_off : h_off + self.crop_height, w_off : w_off + self.crop_width]

        if self.num_crops > 1:
            # Replicate for num_crops
            out = mx.broadcast_to(
                mx.expand_dims(out, axis=1),
                (B, self.num_crops, C, self.crop_height, self.crop_width),
            )
            out = out.reshape(B * self.num_crops, C, self.crop_height, self.crop_width)
        return out

    def output_shape_in(self, input_shape=None):
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(input_shape={self.input_shape}, "
            f"crop_size=[{self.crop_height}, {self.crop_width}], "
            f"num_crops={self.num_crops})"
        )
