@echo off
setlocal
chcp 65001 >nul
cd /d "D:\BOT TRADING\bot-trading"

if not exist "logs" mkdir "logs"

call ".venv\Scripts\activate.bat"

rem === Génère une date et heure sans caractères interdits ===
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do (
    set jour=%%a
    set mois=%%b
    set annee=%%c
)
for /f "tokens=1-2 delims=: " %%a in ('time /t') do (
    set heure=%%a
    set minute=%%b
)
set logfile=logs\kbot_%annee%-%mois%-%jour%_%heure%h%minute%.log

echo [ %date% %time% ] Démarrage de kBot (run unique)...
python -u bot_trading.py >> "%logfile%" 2>&1

echo.
echo Bot terminé. Appuyez sur une touche pour fermer.
pause >nul
