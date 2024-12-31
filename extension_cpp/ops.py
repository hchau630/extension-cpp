import torch
from torch import Tensor

__all__ = ["myexp", "mymuladd", "myadd_out"]


def myexp(a: Tensor) -> Tensor:
    """Performs exp(a) in an efficient fused kernel"""
    return torch.ops.extension_cpp.myexp.default(a)


@torch.library.register_fake("extension_cpp::myexp")
def _(a):
    return torch.empty_like(a)


def mymuladd(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.mymuladd.default(a, b, c)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("extension_cpp::mymuladd")
def _(a, b, c):
    torch._check(a.device == b.device == c.device)
    return torch.empty(
        torch.broadcast_shapes(a.shape, b.shape, c.shape),
        dtype=torch.promote_types(torch.promote_types(a.dtype, b.dtype), c.dtype),
        device=a.device,
        layout=a.layout,
    )


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b, grad_c = None, None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
    if ctx.needs_input_grad[2]:
        grad_c = torch.ops.extension_cpp.mymul.default(grad, torch.ones_like(a))
    return grad_a, grad_b, grad_c


def _setup_context(ctx, inputs, output):
    a, b, _ = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "extension_cpp::mymuladd", _backward, setup_context=_setup_context
)


@torch.library.register_fake("extension_cpp::mymul")
def _(a, b):
    torch._check(a.device == b.device)
    return torch.empty(
        torch.broadcast_shapes(a.shape, b.shape),
        dtype=torch.promote_types(a.dtype, b.dtype),
        device=a.device,
        layout=a.layout,
    )


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.extension_cpp.myadd_out.default(a, b, out)
