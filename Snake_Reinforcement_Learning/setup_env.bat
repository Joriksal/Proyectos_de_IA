@echo on
cd /d %~dp0

echo [1/6] Creando entorno virtual con Python 3.10...
python -m venv venv

echo [2/6] Activando entorno virtual...
call venv\Scripts\activate.bat

echo [3/6] Actualizando pip...
python -m pip install --upgrade pip

echo [4/6] Instalando dependencias desde requirements.txt...
pip install -r requirements.txt

echo [5/6] Instalando el proyecto en modo editable...
pip install -e .

echo [6/6] Entorno configurado correctamente.
echo Para activar manualmente, usa: call venv\Scripts\activate
pause
cls
cmd /k