FROM hughbertlong/utonia-cuda-128-rtx-blackwell@sha256:ee2b31d5203fce5deef20b4994a59e3cc73d74d071928e3070fc2e23936f7230

# The upstream RTX Blackwell image ships torch_scatter's segment_csr kernel
# only for sm_120. B300 reports compute capability 10.3, so retain sm_100 PTX
# for forward-compatible driver JIT instead of relying on an RTX cubin.
ENV TORCH_CUDA_ARCH_LIST="10.0+PTX"
ENV FORCE_CUDA="1"
RUN pip3 install --no-cache-dir --force-reinstall --no-deps \
    --no-build-isolation --no-binary torch-scatter torch-scatter==2.1.2
