#!/bin/bash
conda env update -f ftorf_env.yml
conda activate ftorf
pip install gradient_accumulator --no-deps