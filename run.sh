rm -rf outputs/*
rm -rf eval_output/
python pipeline/run_countright.py --dataset_file "dataset/CoCoCount.json" --step_latent_noise_injection "4"
python evaluation_script.py --images_dir outputs --output_dir eval_output --yolo_weights dataset/yolov9e.pt
