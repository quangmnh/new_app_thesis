#!/bin/bash

python3 freeze_model.py --model=$0 --output=$1
python3 /usr/lib/python3.6/dist-packages/uff/bin/convert_to_uff.py input/$1