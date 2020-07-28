# Datasets

## Setup 
Download raw data
```bash
    sh download_raw_data.sh
```
Data cleaning 
```bash
    python3 datacleaning.py
```
Decompress list of json files
```bash
    sh decompress_list_json_files.sh
```
Join all the files into a single one.
```bash
    sh join_json_files.sh
```
```bash
sudo sh -c 'echo 1 >/proc/sys/vm/drop_caches'
```