@echo off
REM 意图提醒服务启动（v0.1.1）
cd /d %~dp0..\..
set PYTHONPATH=src
.venv_xpu\Scripts\python.exe -m moonbow.intent.service --config config\intent_service.json
