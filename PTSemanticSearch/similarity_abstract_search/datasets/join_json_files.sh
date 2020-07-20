#!/bin/bash

DATA='Data/processed/SemanticScholarData'
# jq -s '{title: .title[]}' Data/processed/SemanticScholarData/pruned_and_clean*.json > Data/processed/foo.json
jq -s '.[0].title = [.[].title | add] | .[0]'  Data/processed/SemanticScholarData/pruned_and_clean*.json > Data/processed/foo.json
