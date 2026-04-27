import sys
import os
import setuptools.dist
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from matplotlib import pyplot as plt

# Add project root to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_PATH, HMSC_HPC_PATH

sys.path.append(HMSC_HPC_PATH)
from hmsc.run_gibbs_sampler import run_gibbs_sampler

path_data = DATA_PATH
modelTypeString = "cov"

samN = 250
thinN = 1
nChains = 1
eagerExecFlag = 0
fp = 32
RS = 1


transient = samN * thinN
verbose = np.maximum(1, int((transient + samN * thinN) / 100))
dtype = np.float32 if fp == 32 else np.float64
tf.config.run_functions_eagerly(eagerExecFlag)
input_path = os.path.join(path_data, "hmsc", "init", "init_%s_chain%.2d.rds" % (modelTypeString, nChains))
output_path = os.path.join(
    path_data,
    "hmsc",
    f"fmTF_localhostname{fp}",
    "TF_%s_chain%.2d_sam%.4d_thin%.4d.rds" % (modelTypeString, nChains, samN, thinN),
)


# tf.random.set_seed(RS)
# np.random.seed(RS)
# run_gibbs_sampler(samples=samN, transient=transient, thin=thinN, verbose=verbose, input=input_path, output=output_path, fse=1, tnlib="tf", profile=0, fp=fp)
# args="""--samples %d --transient %d --thin %d --verbose %d --input %s --output %s --fpb 1 --fse 1 --tnlib tf --profile 0 --fp %d""" % (samN, transient, thinN, verbose, input_path, output_path, fp)
# run_cmd = 'python3 -m hmsc.run_gibbs_sampler ' + args
# %run $run_cmd

run_gibbs_sampler(samN, thinN, transient, verbose, input_path, output_path, dtype=dtype)
