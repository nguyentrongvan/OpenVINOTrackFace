#!/usr/bin/python3
DIR="myenv"
if [ -d "$DIR" ]; then
    cd D:/FaceMoudule
    source venv/Scripts/activate
else
    virtualenv $DIR
    source venv/Scripts/activate
    pip install -r requirements.txt
fi
python run.py