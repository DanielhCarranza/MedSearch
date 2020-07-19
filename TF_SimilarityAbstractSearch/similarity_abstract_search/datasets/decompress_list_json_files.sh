#!/bin/bash
DATA='Data/processed/SemanticScholarData/'
files=$(find $DATA/ -name *.gz)
for fn in $files; do  
  echo $fn 
  gunzip $fn
done 