#!/bin/bash

#python train_classifier.py 0 5000 -k 4 | tee results_nb_i.txt; 
#python train_classifier.py 1 5000 -k 4 | tee results_svm_i.txt;
python train_classifier.py 2 5000 -k 4 | tee results_me_i.txt;
