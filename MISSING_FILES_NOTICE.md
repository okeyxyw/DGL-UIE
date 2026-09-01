# Files still required for a fully runnable public repository

This package contains all code files that were supplied in the current conversation,
plus the generated README and the available 100-point time-series CSV.

The current Python scripts import additional project files that were not supplied here.
Before publishing the repository, add the corresponding original files:

- utils/dataset.py
- utils/metrics.py
- utils/loss_funcs.py
- utils/HVI_transform.py
- utils/CIDNet.py
- utils/utils.py
- utils/uranker/URanker.yaml
- utils/uranker/URanker_ckpt.pth
- utils/uranker/uranker_utils.py
- utils/CIDNet_weight_LOLv2_bestSSIM.pth
- the released DGL-UIE pretrained checkpoint(s)

Also remove or replace machine-specific/hard-coded paths in train.py and test_nr.py.
