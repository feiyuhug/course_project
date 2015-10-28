#!/bin/bash

echo "500" > conf
python train_byLDA.py
python test_model.py

echo "600" > conf
python train_byLDA.py
python test_model.py

echo "700" > conf
python train_byLDA.py
python test_model.py

echo "800" > conf
python train_byLDA.py
python test_model.py

echo "900" > conf
python train_byLDA.py
python test_model.py

