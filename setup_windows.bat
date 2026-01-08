@echo off
echo ===================================================
echo      SAM Audio - Windows Portable Setup
echo ===================================================
echo.
echo [1/4] Creating Python Virtual Environment...
python -m venv venv
call venv\Scripts\activate

echo.
echo [2/4] Installing Core Dependencies...
echo    - Installing PyTorch (CUDA 12.4)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo    - Installing other requirements...
pip install -r requirements.txt

echo.
echo [3/4] Installing SAM Audio (Patched)...
:: Using -e to install in editable mode essentially links the local patched code
pip install -e .

echo.
echo [4/4] Installation Complete!
echo.
echo To run the GUI, use start_gui.bat
echo.
pause
