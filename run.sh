#! /bin/sh
make
python scripts/run_ort.py $@
./a.out $@
#vimdiff output*
