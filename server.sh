#!/usr/bin/python3
DIR="venv"
if [ -d "$DIR" ]; then
    cd D:/OpenVINOTrackFace
    source venv/Scripts/activate
else
    virtualenv $DIR
    source venv/Scripts/activate
    pip install -r requirements.txt
fi
python run.py