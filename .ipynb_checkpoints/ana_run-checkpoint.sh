#!/bin/bash


python tune.py --task cora --backbone gnn --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 10 --random-labels --results-dir ana_results --parallel 8
