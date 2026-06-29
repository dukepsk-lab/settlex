@echo off 
cd /d C:\Users\Administrator\Desktop\settlex 
python -m settlex.cli fetch-data --refresh >> fetch_log.txt 2>&1 
