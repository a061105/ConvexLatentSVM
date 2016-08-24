#!/bin/bash
paste varMap sol > var_sol
python keepNonzero.py var_sol var_sol.nnz
