#!/bin/bash


fine_tune_model_script="/home/shiv/Desktop/projects/model/fine_tune_model_server.py"
main_app_script="/home/shiv/Desktop/projects/main.py"
frontend_app_script="/home/shiv/Desktop/projects/app.py"
venv_path="/home/shiv/Desktop/projects/.venv/bin/activate"




if [ -f "$venv_path" ]; then
    echo " Activating Virtual Environment..."
    source "$venv_path"
else
    echo "Error: Virtual environment not found at $venv_path"
    exit 1
fi
cleanup(){
  echo -e"\n Shutting down all the services"
  kill $(jobs -p) 2>/dev/null
  exit
}


trap cleanup SIGHT


echo "[1/3] Starting Model Server (Port 8000)..."
python "$fine_tune_model_script"&

echo "Waiting for VRAM allocation..."
sleep 10

echo "Starting FastAPI Backend (Port 3000)..."
python "$main_app_script"&


echo "Waiting for FastAPI..."
sleep 3
echo "[3/3] Launching Streamlit Frontend..."
streamlit run "$frontend_app_script"

wait



