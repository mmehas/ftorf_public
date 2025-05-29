#!/bin/bash
conda env update -f ftorf_env.yml
source activate ftorf
pip install gradient_accumulator --no-deps
