# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DarkNet model."""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import ms_function, Tensor


class Bottleneck(nn.Cell):
    # Standard bottleneck
    # ch_in, ch_out, shortcut, groups, expansion
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        c1 = self.cv1(x)
        c2 = self.cv2(c1)
        out = c2
        if self.add:
            out = x + out
        return out


class BottleneckCSP(nn.Cell):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(in_channels=c1, out_channels=c_, kernel_size=1, stride=1, pad_mode='pad', has_bias=False)
        self.cv3 = nn.Conv2d(in_channels=c_, out_channels=c_, kernel_size=1, stride=1, pad_mode='pad', has_bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(num_features=2 * c_, momentum=1-0.03, affine=True, eps=0.001)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(alpha=0.1)
        self.m = nn.SequentialCell([*[Bottleneck(c_, c_, shortcut, 1.0) for _ in range(n)]])

    def construct(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(ops.Concat(1)((y1, y2)))))


class SPP(nn.Cell):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)

        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, pad_mode='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, pad_mode='same')
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, pad_mode='same')
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.cv1(x)
        m1 = self.maxpool1(c1)
        m2 = self.maxpool2(c1)
        m3 = self.maxpool3(c1)
        c4 = self.concat((c1, m1, m2, m3))
        c5 = self.cv2(c4)
        return c5


class Focus(nn.Cell):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, act)
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        input = self.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]])
        c1 = self.conv(input)
        return c1


def auto_pad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Cell):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None,
                 dilation=1,
                 alpha=0.1,
                 momentum=0.97,
                 eps=1e-3,
                 pad_mode="same",
                 act=True):  # ch_in, ch_out, kernel, stride, padding
        super(Conv, self).__init__()
        self.padding = auto_pad(k, p)
        self.pad_mode = None
        if self.padding == 0:
            self.pad_mode = 'same'
        elif self.padding == 1:
            self.pad_mode = 'pad'
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            padding=self.padding,
            pad_mode=self.pad_mode,
            has_bias=False)
        self.bn = nn.BatchNorm2d(c2, affine=True, momentum=momentum, eps=eps)
        # self.act = nn.LeakyReLU() if act is True else (
        #     act if isinstance(act, nn.Cell) else ops.Identity())
        try:
            self.act = Hardswish() if act else ops.Identity()
        except:
            self.act = ops.Identity()

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))


class Hardswish(nn.Cell):  # export-friendly version of nn.Hardswish()
    @ms_function
    def construct(self, x):
        return x * hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


def hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0) -> Tensor:
    # from torch.nn.functional.hardtanh
    r"""
    hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor
    Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
    details.
    """
    max_tensor = max_val*ops.OnesLike()(input)  # max_tensor = Tensor(max_val*np.ones_like(input.asnumpy()))
    result = ops.Minimum()(input, max_tensor)
    min_tensor = min_val*ops.OnesLike()(input)  # min_tensor = Tensor(min_val*np.ones_like(input.asnumpy()))
    result = ops.Maximum()(result, min_tensor)
    return result


class YOLOPBackbone(nn.Cell):
    def __init__(self):
        super(YOLOPBackbone, self).__init__()  # [3, 32, 64, 128, 256, 512, 1]
        self.focus = Focus(3, 32, k=3, s=1)  # 0
        self.conv1 = Conv(32, 64, k=3, s=2)  # 1
        self.csp2 = BottleneckCSP(64, 64, n=1)  # 2
        self.conv3 = Conv(64, 128, k=3, s=2)  # 3
        self.csp4 = BottleneckCSP(128, 128, n=3)  # 4
        self.conv5 = Conv(128, 256, k=3, s=2)  # 5
        self.csp6 = BottleneckCSP(256, 256, n=3)  # 6
        self.conv7 = Conv(256, 512, k=3, s=2)  # 7
        self.spp8 = SPP(512, 512, k=[5, 9, 13])  # 8
        self.csp9 = BottleneckCSP(512, 512, n=1, shortcut=False)  # 9

    def construct(self, x):
        """construct method"""
        e0 = self.focus(x)
        e1 = self.conv1(e0)
        e2 = self.csp2(e1)
        e3 = self.conv3(e2)
        # out
        e4 = self.csp4(e3)  # 4
        e5 = self.conv5(e4)
        # out
        e6 = self.csp6(e5)  # 6
        e7 = self.conv7(e6)
        e8 = self.spp8(e7)
        # out
        e9 = self.csp9(e8)  # 9
        return e4, e6, e9
