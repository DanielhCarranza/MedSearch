#!/bin/bash

DATA_PATH='Data/processed/SemanticScholarData/'
# mkdir $DATA_PATH 
# rm Data/processed/paper_set.txt
# wget https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-05-27/manifest.txt -P 'Data/processed/'  

src='https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-04-10/'
file_names=$(grep s2-corpus Data/processed/manifest.txt)

for fn in $file_names; do
    jsonfn=$(echo $fn | cut -d'.' -f 1)'.json'
    echo $DATA_PATH$jsonfn
    wget -qO- $src$fn | gunzip -c - > 'Data/processed/corpus.json'  
    cat Data/processed/corpus.json | \
    jq 'select(.fieldsOfStudy== ["Biology"] or ["Medicine"]) | select(.pmid !="" or .sources=="Medline") | {id:.id,  paperAbstract: .paperAbstract, title:.title, journalName:.journalName, venue:.venue, totalCitation:(.outCitations + .inCitations)} | select(.totalCitation !=[]) |  @json' > $DATA_PATH$jsonfn
    jq ' fromjson | .id ' $DATA_PATH$jsonfn >> Data/processed/paper_set.txt
    rm Data/processed/corpus.json
done
