#!/bin/bash


python tune.py --task agedb --backbone cnn --model cnn vib nib svib ceb dvcca fgib --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 10

