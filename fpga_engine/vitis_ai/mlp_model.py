# mlp_model.py — PyTorch definition of the same 16 -> 32 (ReLU) -> 8 MLP
# that ml_kernel.cpp implements as a hand-written, fully-pipelined HLS
# kernel, so the Vitis AI flow in vai_compile_flow.sh quantizes/compiles
# the identical network shape instead of a stand-in.
#
# Vitis AI's quantizer/compiler only accept a model in a supported
# framework format (PyTorch/TensorFlow) plus a calibration dataset -- they
# have no path that takes raw C++ arrays like mlp_int8_ref.cpp's, which is
# why this file exists as a separate, framework-specific definition rather
# than reusing that file directly. Weight layout matches ml_kernel.cpp's
# w1[16][32]/w2[32][8] (row = input index, col = output index) so a
# state_dict exported from here can be loaded verbatim into ml_kernel.cpp's
# test harness for an apples-to-apples comparison once both actually run.
#
# TODO: requires PyTorch + the Vitis AI quantizer (vai_q_pytorch) --
# neither installed locally. Unrun.

import torch
import torch.nn as nn

K_INPUTS = 16
K_HIDDEN = 32
K_OUTPUTS = 8


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # bias-free, matching ml_kernel.cpp (no bias terms in either layer)
        self.fc1 = nn.Linear(K_INPUTS, K_HIDDEN, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(K_HIDDEN, K_OUTPUTS, bias=False)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def calibration_loader(n_samples=200, seed=42):
    """Same shape/seed/distribution as mlp_int8_ref.cpp's calibration set
    (200 samples, U(-1,1)) so the two quantization schemes are calibrated
    on comparable data, not just comparable network shapes."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_samples):
        yield torch.empty(1, K_INPUTS).uniform_(-1, 1, generator=g)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = TinyMLP().eval()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total multiplies (== param count, no bias): {total_params}")
