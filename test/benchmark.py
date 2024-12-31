import argparse
import math
import time

import torch

import extension_cpp

TIME_SCALES = {"s": 1, "ms": 1000, "us": 1000000}


def exp(a):
    return torch.exp(a)


def muladd(a, b, c):
    return a * b + c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("func", choices=["muladd", "exp"])
    parser.add_argument("example", choices=["py", "cpp", "cuda"])
    parser.add_argument("-n", type=int, default=1 << 20)
    parser.add_argument("-r", "--runs", type=int, default=100)
    parser.add_argument("--scale", choices=["s", "ms", "us"], default="us")
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-d", "--double", action="store_true")
    options = parser.parse_args()

    if options.example == "py":
        func = globals()[options.func]
    else:
        func = getattr(extension_cpp.ops, f"my{options.func}")
    if options.example == "cuda":
        options.cuda = True

    device = torch.device("cuda") if options.cuda else torch.device("cpu")
    dtype = torch.float64 if options.double else torch.float32

    kwargs = {"dtype": dtype, "device": device, "requires_grad": True}

    if options.func == "exp":
        args = (torch.randn((options.n,), **kwargs),)
    else:
        args = (
            torch.randn((options.n,), **kwargs),
            torch.randn((options.n,), **kwargs),
            torch.randn((options.n,), **kwargs),
        )

    forward_min = math.inf
    forward_time = 0
    backward_min = math.inf
    backward_time = 0
    for _ in range(options.runs):
        for t in args:
            if isinstance(t, torch.Tensor):
                t.grad = None

        start = time.time()
        out = func(*args)
        elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed

        start = time.time()
        out.sum().backward()
        elapsed = time.time() - start
        backward_min = min(backward_min, elapsed)
        backward_time += elapsed

    scale = TIME_SCALES[options.scale]
    forward_min *= scale
    backward_min *= scale
    forward_average = forward_time / options.runs * scale
    backward_average = backward_time / options.runs * scale

    print(
        "Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}".format(
            forward_min, forward_average, backward_min, backward_average, options.scale
        )
    )


if __name__ == "__main__":
    main()
