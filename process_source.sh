#!/bin/sh
# usage: sh process_source.sh <input.csv with file column> <tag>
# -> data/faces<tag>.csv + data/feats_<backbone><tag>.npy (face x4, whole photo, upper body)
set -e
PY=./.venv/Scripts/python
$PY -W ignore crop_faces.py "$1" "data/faces$2.csv"
$PY -W ignore embed_multi.py --csv "data/faces$2.csv" --tag "$2" facenet clipL clipH siglip2
$PY -W ignore embed_context.py "data/faces$2.csv" "$2"
echo "done $2"
