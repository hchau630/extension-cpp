# C++/CUDA Extensions in PyTorch

An example of writing a C++/CUDA extension for PyTorch. See
[here](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html) for the accompanying tutorial.
This repo demonstrates how to write an example `extension_cpp.ops.mymuladd`
custom op that has both custom CPU and CUDA kernels.

# Ho Yin's notes
The examples in this repo work with PyTorch 2.4+.

To get it to work on Axon with an A40 GPU, I performed the following steps:
1) `conda create -n extension-cpp python=3.11`
2) `pip install -r requirements.txt`
3) `conda install nvidia/label/cuda-12.4.1::cuda-toolkit`
4) `ml gcc/10.4`
4) `export CPATH=/home/hc3190/.conda/envs/extension-cpp/targets/x86_64-linux/include/:$CPATH`
5) `pip install .`

To test:
```
python test/test_extension.py
```

## Authors

[Peter Goldsborough](https://github.com/goldsborough), [Richard Zou](https://github.com/zou3519)
