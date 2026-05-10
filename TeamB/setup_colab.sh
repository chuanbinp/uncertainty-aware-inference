# setup_colab.sh - run once in Colab before notebooks
pip install -q \
  transformers==4.51.3 accelerate==1.13.0 \
  auto-gptq==0.7.1 autoawq==0.2.9 bitsandbytes==0.49.2 \
  optimum==1.24.0 netcal==1.3.5 wandb datasets
