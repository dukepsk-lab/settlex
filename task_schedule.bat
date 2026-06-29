echo @echo off > C:\Users\Administrator\Desktop\settlex\fetch_data.bat
echo cd /d C:\Users\Administrator\Desktop\settlex >> C:\Users\Administrator\Desktop\settlex\fetch_data.bat
echo python -m settlex.cli fetch-data --refresh ^>^> fetch_log.txt 2^>^&1 >> C:\Users\Administrator\Desktop\settlex\fetch_data.bat

echo @echo off > C:\Users\Administrator\Desktop\settlex\run_bot.bat
echo cd /d C:\Users\Administrator\Desktop\settlex >> C:\Users\Administrator\Desktop\settlex\run_bot.bat
echo python -m settlex.cli run --execute ^>^> run_log.txt 2^>^&1 >> C:\Users\Administrator\Desktop\settlex\run_bot.bat

schtasks /create /tn "Settlex_FetchData" /tr "C:\Users\Administrator\Desktop\settlex\fetch_data.bat" /sc daily /st 18:00 /rl HIGHEST /f

schtasks /create /tn "Settlex_RunBot" /tr "C:\Users\Administrator\Desktop\settlex\run_bot.bat" /sc daily /st 10:05 /rl HIGHEST /f