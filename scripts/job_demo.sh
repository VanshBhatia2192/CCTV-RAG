export GRADIO_TEMP_DIR=$PWD/outputs/gradio_tmp; mkdir -p $GRADIO_TEMP_DIR
export GRADIO_TEMP_DIR=$PWD/outputs/gradio_tmp; mkdir -p $GRADIO_TEMP_DIR
export GRADIO_TEMP_DIR=$PWD/outputs/gradio_tmp; mkdir -p $GRADIO_TEMP_DIR
export GRADIO_SERVER_NAME=0.0.0.0 GRADIO_ANALYTICS_ENABLED=False
(while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> outputs/logs/gpu_util.log; sleep 5; done) &
echo "UI on $(hostname):7860"
python -u -m src.ui.app
