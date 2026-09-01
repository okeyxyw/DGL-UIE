import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.HVI_transform import RGB_HVI

class SepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channel,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


## HIN Block
class BasicBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, relu_slope=0.1):
        super(BasicBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = SepConv(in_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = SepConv(out_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):
        out = self.conv_1(x)

        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out = out + self.identity(x)
        return out


class WaveletEnhanceBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer("haar_kernel", kernel)

        self.fuse = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False)
        self.post = SepConv(channels, channels, kernel_size=3, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        dwt = F.conv2d(x, self.haar_kernel, stride=2, groups=C)

        fea = self.fuse(dwt)  # -> [B, C, H//2, W//2]
        fea = self.post(fea)  # -> [B, C, H//2, W//2]

        out = F.interpolate(fea, size=(H, W), mode="bilinear", align_corners=False)

        return out


class GetGradient(nn.Module):
    def __init__(self, dim=3, mode="sobel"):
        super(GetGradient, self).__init__()
        self.dim = dim
        self.mode = mode
        if mode == "sobel":
            # sobel filter
            kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

            kernel_y = (
                torch.tensor(kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            kernel_x = (
                torch.tensor(kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )

            self.register_buffer("kernel_y", kernel_y.repeat(self.dim, 1, 1, 1))
            self.register_buffer("kernel_x", kernel_x.repeat(self.dim, 1, 1, 1))
        elif mode == "laplacian":
            kernel_laplace = [[0.25, 1, 0.25], [1, -5, 1], [0.25, 1, 0.25]]
            kernel_laplace = (
                torch.tensor(kernel_laplace, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            self.register_buffer(
                "kernel_laplace", kernel_laplace.repeat(self.dim, 1, 1, 1)
            )

    def forward(self, x):
        if self.mode == "sobel":
            grad_x = F.conv2d(x, self.kernel_x, padding=1, groups=self.dim)
            grad_y = F.conv2d(x, self.kernel_y, padding=1, groups=self.dim)

            grad_magnitude = torch.sqrt(
                torch.pow(grad_x, 2) + torch.pow(grad_y, 2) + 1e-6
            )
        elif self.mode == "laplacian":
            grad_magnitude = F.conv2d(
                x, self.kernel_laplace, padding=1, groups=self.dim
            )
            grad_magnitude = torch.abs(grad_magnitude)  # magnitude only

        return grad_magnitude


class SGFB(nn.Module):
    def __init__(self, feature_channels=48):
        super(SGFB, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.frdb1 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.frdb2 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.get_gradient = GetGradient(feature_channels, mode="sobel")
        self.conv_grad = nn.Sequential(
            SepConv(feature_channels, feature_channels, kernel_size=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        grad = self.get_gradient(x)
        grad = self.conv_grad(grad)
        x = self.frdb1(x)
        alpha = torch.sigmoid(self.alpha)
        x = alpha * grad * x + (1 - alpha) * x
        x = self.frdb2(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, feature_channels=48):
        super(BasicLayer, self).__init__()
        self.fwawb = DAFFB(feature_channels)
        self.sgfb = SGFB(feature_channels)

    def forward(self, x):
        res = x
        x = self.fwawb(x) + x
        x = self.sgfb(x)
        return 0.5 * x + 0.5 * res


class HVIBCCWhiteBalance(nn.Module):
    """
    HVI-BCC-WBP
    HVI Brightness-Contrast-Chroma
    Decoupled White-Balance Prior

    Pipeline:
        RGB
         ↓
        Gray-World
         ↓
        RGB -> HVI
         ↓
        ┌─────────────┬─────────────┐
        │             │             │
    Brightness     Contrast       Chroma
       I branch      I branch      H/V branch
        │             │             │
        └─────────────┴─────────────┘
                      ↓
                    HVI'
                      ↓
                  HVI -> RGB

    All final convolutions are zero initialized.

    Therefore at initialization:

        delta_b = 0
        delta_c = 0
        delta_hv = 0

    => HVI output == Gray-World output

    This preserves the original WWE-UIE prior
    at the beginning of training.
    """

    def __init__(
        self,
        hidden=16,
        brightness_scale=0.20,
        contrast_scale=0.20,
        chroma_scale=0.15,
        eps=1e-6
    ):
        super().__init__()

        self.eps = eps

        self.brightness_scale = brightness_scale
        self.contrast_scale = contrast_scale
        self.chroma_scale = chroma_scale

        # =====================================================
        # Original WWE-UIE white-balance prior
        # =====================================================

        self.gray_world = GrayWorldRetinex(
            eps=eps
        )

        # =====================================================
        # HVI transform
        # =====================================================

        self.hvi = RGB_HVI()

        # If HVI implementation contains learnable
        # density parameter, keep it fixed here.
        if hasattr(self.hvi, "density_k"):
            self.hvi.density_k.requires_grad_(False)

        # =====================================================
        # 1. Brightness branch
        #
        # I -> delta brightness
        # =====================================================

        self.brightness_branch = nn.Sequential(

            nn.Conv2d(
                1,
                hidden,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                1,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )

        # =====================================================
        # 2. Contrast branch
        #
        # I -> contrast gain
        # =====================================================

        self.contrast_branch = nn.Sequential(

            nn.Conv2d(
                1,
                hidden,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                1,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )

        # =====================================================
        # 3. Chroma branch
        #
        # [H,V] -> chromatic correction
        # =====================================================

        self.chroma_branch = nn.Sequential(

            nn.Conv2d(
                2,
                hidden,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                2,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )

        # =====================================================
        # Identity-preserving initialization
        #
        # Important:
        #
        # the last conv = 0
        #
        # therefore the model starts exactly from
        # the original Gray-World prior.
        # =====================================================

        self._zero_init_last_conv(
            self.brightness_branch
        )

        self._zero_init_last_conv(
            self.contrast_branch
        )

        self._zero_init_last_conv(
            self.chroma_branch
        )

    @staticmethod
    def _zero_init_last_conv(branch):

        last = branch[-1]

        nn.init.zeros_(
            last.weight
        )

        if last.bias is not None:
            nn.init.zeros_(
                last.bias
            )

    def forward(self, x):

        # =====================================================
        # 1. Original Gray-World prior
        # =====================================================

        x_gw = self.gray_world(x)

        x_gw = torch.clamp(
            x_gw,
            0.0,
            1.0
        )

        # =====================================================
        # 2. RGB -> HVI
        # =====================================================

        hvi = self.hvi.HVIT(
            x_gw
        )

        h = hvi[:, 0:1, :, :]
        v = hvi[:, 1:2, :, :]
        intensity = hvi[:, 2:3, :, :]

        # =====================================================
        # 3. Brightness decoupling
        # =====================================================

        delta_b = torch.tanh(
            self.brightness_branch(
                intensity
            )
        )

        delta_b = (
            self.brightness_scale
            *
            delta_b
        )

        # =====================================================
        # 4. Contrast decoupling
        # =====================================================

        delta_c = torch.tanh(
            self.contrast_branch(
                intensity
            )
        )

        contrast_gain = (
            1.0
            +
            self.contrast_scale
            *
            delta_c
        )

        # =====================================================
        # 5. Brightness + Contrast reconstruction
        #
        # Inspired by:
        #
        # L' = (L + delta_L) * G_c
        #
        # =====================================================

        intensity_new = (
            intensity
            +
            delta_b
        )

        intensity_new = (
            intensity_new
            *
            contrast_gain
        )

        intensity_new = torch.clamp(
            intensity_new,
            0.0,
            1.0
        )

        # =====================================================
        # 6. Chroma decoupling
        # =====================================================

        chroma = torch.cat(
            [h, v],
            dim=1
        )

        delta_chroma = torch.tanh(
            self.chroma_branch(
                chroma
            )
        )

        chroma_new = (
            chroma
            +
            self.chroma_scale
            *
            delta_chroma
        )

        h_new = chroma_new[
            :, 0:1, :, :
        ]

        v_new = chroma_new[
            :, 1:2, :, :
        ]

        # Prevent numerical explosion.
        #
        # If your HVI implementation uses a different
        # valid H/V range, this clamp can be relaxed.
        h_new = torch.clamp(
            h_new,
            -1.0,
            1.0
        )

        v_new = torch.clamp(
            v_new,
            -1.0,
            1.0
        )

        # =====================================================
        # 7. HVI reconstruction
        # =====================================================

        hvi_new = torch.cat(
            [
                h_new,
                v_new,
                intensity_new
            ],
            dim=1
        )

        # =====================================================
        # 8. HVI -> RGB
        # =====================================================

        x_hvi = self.hvi.PHVIT(
            hvi_new
        )

        x_hvi = torch.clamp(
            x_hvi,
            0.0,
            1.0
        )

        return x_hvi
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class GrayWorldRetinex(nn.Module):
    def __init__(self, eps=1e-6):
        super(GrayWorldRetinex, self).__init__()
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        mean = x.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        gray_mean = mean.mean(dim=1, keepdim=True)  # [B, 1, 1, 1]
        gain = gray_mean / (mean + self.eps)
        x = x * gain  # white balance
        x_log = torch.log(x + self.eps)
        x_log = x_log - x_log.mean(dim=(2, 3), keepdim=True)
        x_out = torch.exp(x_log)
        x_min = x_out.amin(dim=(-2, -1), keepdim=True)
        x_max = x_out.amax(dim=(-2, -1), keepdim=True)
        x_out = (x_out - x_min) / (x_max - x_min + self.eps)
        return x_out

class DAFFB(nn.Module):
    """
    Degradation-Adaptive Fourier Frequency Block

    1. FFT amplitude-phase decoupling
    2. Adaptive low/high frequency routing
    3. Dynamic frequency cutoff
    4. Preserve phase information
    5. Zero-initialized residual frequency modulation
    """

    def __init__(
        self,
        channels,
        reduction=4,
        cutoff_min=0.18,
        cutoff_max=0.48
    ):
        super().__init__()

        self.channels = channels
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

        hidden = max(
            channels // reduction,
            8
        )

        # ==========================================
        # 1. Adaptive frequency router
        #
        # Output:
        # 2*C -> low/high weights
        # 1   -> dynamic cutoff
        # ==========================================

        self.router_pool = nn.AdaptiveAvgPool2d(1)

        self.router = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=True
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                2 * channels + 1,
                kernel_size=1,
                bias=True
            )
        )

        # Initial low/high weights:
        #
        # softmax([0,0]) = [0.5,0.5]
        #
        # cutoff is also initialized
        # near the middle of the range.
        nn.init.zeros_(
            self.router[-1].weight
        )

        nn.init.zeros_(
            self.router[-1].bias
        )

        # ==========================================
        # 2. Low-frequency enhancement
        # ==========================================

        self.low_refine = nn.Sequential(

            SepConv(
                channels,
                channels,
                kernel_size=3,
                bias=False
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        # ==========================================
        # 3. High-frequency enhancement
        # ==========================================

        self.high_refine = nn.Sequential(

            SepConv(
                channels,
                channels,
                kernel_size=3,
                bias=False
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        # ==========================================
        # 4. Residual frequency strength
        #
        # beta = 0 initially
        #
        # => amplitude is unchanged
        # ==========================================

        self.beta = nn.Parameter(
            torch.zeros(1)
        )

        # ==========================================
        # 5. Spatial refinement
        # ==========================================

        self.post = SepConv(
            channels,
            channels,
            kernel_size=3,
            bias=False
        )

    @staticmethod
    def _radius_map(
        h,
        w,
        device,
        dtype
    ):
        """
        Generate normalized radial frequency map.

        Center frequency is represented according
        to FFT's original unshifted coordinates.
        """

        fy = torch.fft.fftfreq(
            h,
            d=1.0,
            device=device,
            dtype=dtype
        )

        fx = torch.fft.fftfreq(
            w,
            d=1.0,
            device=device,
            dtype=dtype
        )

        yy, xx = torch.meshgrid(
            fy,
            fx,
            indexing="ij"
        )

        # Normalize frequency radius
        radius = torch.sqrt(
            (xx / 0.5) ** 2
            +
            (yy / 0.5) ** 2
        )

        radius = radius / (
            2.0 ** 0.5
        )

        return radius.unsqueeze(
            0
        ).unsqueeze(
            0
        )

    def forward(self, x):

        B, C, H, W = x.shape

        # ==========================================
        # 1. FFT
        # ==========================================

        spectrum = torch.fft.fft2(
            x,
            dim=(-2, -1),
            norm="ortho"
        )

        # ==========================================
        # 2. Amplitude
        # ==========================================

        amplitude = torch.abs(
            spectrum
        )

        # ==========================================
        # 3. Phase-preserving representation
        #
        # Instead of using angle(),
        # directly obtain complex unit phase.
        # ==========================================

        phase_unit = spectrum / (
            amplitude + 1e-8
        )

        # log amplitude makes optimization
        # more stable
        log_amp = torch.log1p(
            amplitude
        )

        # ==========================================
        # 4. Degradation-aware routing
        # ==========================================

        route = self.router(
            self.router_pool(x)
        )

        # ------------------------------
        # low / high frequency weights
        # ------------------------------

        band_logits = route[
            :, :2 * C
        ]

        band_logits = band_logits.view(
            B,
            C,
            2,
            1,
            1
        )

        band_weight = torch.softmax(
            band_logits,
            dim=2
        )

        w_low = band_weight[
            :, :, 0
        ]

        w_high = band_weight[
            :, :, 1
        ]

        # ==========================================
        # 5. Dynamic frequency cutoff
        # ==========================================

        cutoff_logit = route[
            :,
            2 * C:2 * C + 1
        ]

        cutoff = (
            self.cutoff_min
            +
            (
                self.cutoff_max
                -
                self.cutoff_min
            )
            *
            torch.sigmoid(
                cutoff_logit
            )
        )

        # ==========================================
        # 6. Frequency coordinate
        # ==========================================

        radius = self._radius_map(
            H,
            W,
            x.device,
            x.dtype
        )

        # ==========================================
        # 7. Soft low/high frequency masks
        # ==========================================

        low_mask = torch.exp(
            -
            (
                radius
                /
                (
                    cutoff
                    +
                    1e-6
                )
            ) ** 4
        )

        high_mask = (
            1.0
            -
            low_mask
        )

        # ==========================================
        # 8. Low / high decomposition
        # ==========================================

        low_amp = (
            log_amp
            *
            low_mask
        )

        high_amp = (
            log_amp
            *
            high_mask
        )

        # ==========================================
        # 9. Frequency-specific enhancement
        # ==========================================

        delta_low = self.low_refine(
            low_amp
        )

        delta_low = (
            delta_low
            *
            low_mask
        )

        delta_high = self.high_refine(
            high_amp
        )

        delta_high = (
            delta_high
            *
            high_mask
        )

        # ==========================================
        # 10. Adaptive frequency fusion
        # ==========================================

        delta = (
            w_low
            *
            delta_low
            +
            w_high
            *
            delta_high
        )

        # ==========================================
        # 11. Safe amplitude modulation
        #
        # beta starts from 0:
        #
        # amplitude_new = amplitude
        # ==========================================

        strength = torch.tanh(
            self.beta
        )

        amplitude_new = (
            amplitude
            *
            torch.exp(
                strength
                *
                torch.tanh(
                    delta
                )
            )
        )

        # ==========================================
        # 12. Reconstruct spectrum
        #
        # amplitude changes,
        # phase remains unchanged
        # ==========================================

        spectrum_new = (
            phase_unit
            *
            amplitude_new
        )

        # ==========================================
        # 13. IFFT
        # ==========================================

        recon = torch.fft.ifft2(
            spectrum_new,
            dim=(-2, -1),
            norm="ortho"
        ).real

        # ==========================================
        # 14. Residual frequency enhancement
        #
        # Important:
        #
        # BasicLayer already performs:
        #
        # x = self.fwawb(x) + x
        #
        # so only return the residual here.
        # ==========================================

        residual = (
            recon - x
        )

        residual = self.post(
            residual
        )

        return residual
class myModel(nn.Module):
    def __init__(self, in_channels=3, feature_channels=32, use_white_balance=False):
        super(myModel, self).__init__()
        self.use_white_balance = use_white_balance
        if self.use_white_balance:
            self.wb =  HVIBCCWhiteBalance(
                        hidden=16,
                        brightness_scale=0.20,
                        contrast_scale=0.20,
                        chroma_scale=0.15
                    )
            self.alpha = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)

        self.first = nn.Conv2d(
            in_channels, feature_channels, kernel_size=3, stride=1, padding=1
        )
        self.encoder1 = BasicLayer(feature_channels)
        self.down1 = Downsample(feature_channels)
        self.encoder2 = BasicLayer(feature_channels * 2**1)
        self.down2 = Downsample(feature_channels * 2**1)
        self.bottleneck = BasicLayer(feature_channels * 2**2)
        self.up1 = Upsample(feature_channels * 2**2)
        self.decoder1 = BasicLayer(feature_channels * 2**1)
        self.up2 = Upsample(feature_channels * 2**1)
        self.decoder2 = BasicLayer(feature_channels)
        self.out = nn.Conv2d(
            feature_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        res = x
        if self.use_white_balance:
            alpha = torch.sigmoid(self.alpha)
            x = alpha * self.wb(x) + (1 - alpha) * x
        x1 = self.first(x)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))
        x = self.up1(x3) + x2
        x = self.decoder1(x)
        x = self.up2(x) + x1
        x = self.decoder2(x)
        out = self.out(x) + res
        return out


if __name__ == "__main__":
    dummy_img = torch.rand(1, 3, 128, 128)
    model = myModel()
    output_img, output_img_ds, learned_weight_map = model(dummy_img)
    print("Output image shape:", output_img.shape)
    print("Learned weight map shape:", learned_weight_map.shape)
