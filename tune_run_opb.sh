#!/bin/bash

python tune.py --task agedb --backbone mlp --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8

python rebuild_tune_html.py --task agedb --backbone mlp --model mlp vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8




python tune.py --task agedb --backbone cnn --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8

python rebuild_tune_html.py --task agedb --backbone cnn --model cnn vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8





python tune.py --task housing --backbone mlp --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8

python rebuild_tune_html.py --task housing --backbone mlp --model mlp vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8





python tune.py --task mnist --backbone mlp --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8

python rebuild_tune_html.py --task mnist --backbone mlp --model mlp vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8




python tune.py --task mnist --backbone cnn --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8

python rebuild_tune_html.py --task mnist --backbone cnn --model cnn vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8




python tune.py --task imagenet100 --backbone mlp --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8

python rebuild_tune_html.py --task imagenet100 --backbone mlp --model mlp vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8





python tune.py --task imagenet100 --backbone cnn --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8


python rebuild_tune_html.py --task imagenet100 --backbone cnn --model cnn vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8




python tune.py --task agnews --backbone rnn --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8

python rebuild_tune_html.py --task agnews --backbone rnn --model rnn vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8





python tune.py --task imdb --backbone rnn --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 4

python rebuild_tune_html.py --task imdb --backbone rnn --model rnn vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 4




python tune.py --task stsb --backbone rnn --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8

python rebuild_tune_html.py --task stsb --backbone rnn --model rnn vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8





python tune.py --task cora --backbone gnn --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8

python rebuild_tune_html.py --task cora --backbone gnn --model gcn vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8





python tune.py --task zinc --backbone gnn --model opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8


python rebuild_tune_html.py --task zinc --backbone gnn --model gcn vib nib svib ceb dvcca fgib opb --beta 10 1 0.1 0.01 0.001 0.0001 --anchor-scale 1 2 4 6 8 10 12 14 16 --parallel 8



