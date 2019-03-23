#!/bin/bash
# TwinGAN and face swap
cd /Main/
conda env create -n cloned_env -f spec-file.txt
python face_replacer.py
python image_translation_infer.py \ --model_path="256/" --image_hw=256 --input_tensor_name="sources_ph" --output_tensor_name="custom_generated_t_style_source:0" --input_image_path="./Main/temp/" --output_image_path="./Main/output/"
python face_replacer_2.py

# Running CartoonGAN
cd cartoonGAN-Test-PyTorch-Torch-master/cartoonGAN-Test-PyTorch-Torch-master
python test.py --input_dir './input' --output_dir './output_torch' --style Hosoda --gpu 0
