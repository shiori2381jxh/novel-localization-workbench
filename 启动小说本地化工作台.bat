@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo 正在准备小说本地化工作台，首次启动需要安装桌面组件……
  python -m venv .venv
  if errorlevel 1 goto :failed
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
)

start "" ".venv\Scripts\pythonw.exe" app.py
exit /b 0

:failed
echo.
echo 安装或启动失败，请确认已经安装 Python 3.10 或更高版本并可以连接网络。
pause
exit /b 1
