#!/bin/bash
export PYTHONPATH=$(pwd)/..
bash run_complex.sh train ComplEx FB15k-237 0 0 1024 256 128 200.0 0.5 0.001 100000 16 0.000005 -de -dr -seed 0
bash run_complex.sh train ComplEx wn18rr 0 0 512 1024 128 200.0 0.0 0.002 80000 8 0.000001 -de -dr -seed 0
bash run_complex.sh train ComplEx YAGO3-10 0 0 512 1024 128 200.0 0.25 0.001 150000 8 0.0000005 -de -dr -seed 0

bash run.sh train RotatE FB15k-237 0 0 1024 256 128 6.0 1.0 0.00005 100000 16 -de -seed 0
bash run.sh train RotatE wn18rr 0 0 512 1024 128 3 1.0 0.00005 80000 8 -de -seed 0
bash run.sh train RotatE YAGO3-10 0 0 1024 400 128 24 0.75 0.0002 150000 4 -de -seed 0