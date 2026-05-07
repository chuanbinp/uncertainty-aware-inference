# #!/usr/bin/env bash
# # Run AFTER: conda env create -f environment.yml && conda activate quant-eval-cuda13
# #
# # Only needed if autoawq fails to install from the env file
# # (it sometimes needs torch at build time too).
# set -e
 
# python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.version.cuda}')"
 
# echo "Installing quantization backends..."
# python -m pip install --upgrade optimum
# python -m pip install gptqmodel>=6.0.0 --no-build-isolation
# python -m pip install autoawq>=0.2.0 --no-build-isolation
# python -m pip install bitsandbytes>=0.43.0
# echo "Done."