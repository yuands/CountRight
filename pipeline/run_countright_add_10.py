import sys

sys.path.append('.')
from utils.counting_words_extract import find_nummod, word2number
from pipeline.mask_extraction.extract_mask import spatial_prior_predictor
# from pipeline.module.showcam import visualize_and_save  # disabled to save memory
# from pipeline.module.fenlei import visualize_and_save_masks  # disabled to save memory
from pipeline.module.latent_noise_injection import apply_latent_partition
import torch
import argparse
import json
import numpy as np
import random
import yaml
import diffusers
from diffusers.utils.torch_utils import randn_tensor
import os
from pipeline.self_counting_sdxl_pipeline import SelfCountingSDXLPipeline
from utils.generate_random_masks import generate_random_masks_factory, show_mask, show_mask_list
from tqdm import tqdm


def read_yaml(file_path):
    with open(file_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data


def set_seed(seed: int):
    """
    Set the seed for reproducibility in PyTorch, NumPy, and Python's random module.

    Parameters:
    - seed (int): The seed value to use for all random number generators.
    """
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # Make CuDNN backend deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def init_sdxl_model(config):
    sdxl_pipe = SelfCountingSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True,
        torch_dtype=torch.float16, variant="fp16", use_onnx=False
    )

    device = torch.device(config["pipeline"]["device"])
    sdxl_pipe.to(device)

    sdxl_pipe.counting_config = config['counting_model']

    if config['counting_model']['use_ddpm']:
        print('Using DDPM as scheduler.')
        sdxl_pipe.scheduler = diffusers.DDPMScheduler.from_config(sdxl_pipe.scheduler.config)

    return sdxl_pipe, device


def run_counting_pipeline_corrected_masks(sdxl_pipe, prompt, generator, object_masks, latents, config,
                                          step_latent_noise_injection=0):
    out = sdxl_pipe(prompt=[prompt],
                    num_inference_steps=config["counting_model"]["num_inference_steps"],
                    perform_counting=True,
                    desired_mask=object_masks,
                    generator=generator,
                    latents=latents,
                    step_latent_noise_injection=step_latent_noise_injection).images

    image = out[0]
    return image


def run_pipeline(prompt_objects, config, phase1_type, phase2_type, step_latent_noise_injection=0):
    # make directory
    out_dir = config["pipeline"]["output_path"]
    os.makedirs(f'{out_dir}', exist_ok=True)

    metadata_json = []

    if os.path.isfile(f"{out_dir}/metadata.json"):
        with open(f"{out_dir}/metadata.json", "r") as f:
            metadata_json = json.load(f)

    # Initialize SDXL model
    sdxl_pipe, device = init_sdxl_model(config)

    for prompt_object in prompt_objects:
        prompt = prompt_object['prompt']
        seed = prompt_object['seed']

        # Create latents
        set_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        shape = (1, sdxl_pipe.unet.config.in_channels, 128, 128)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float16)

        # Extract object + number
        required_object_num = prompt_object['int_number']
        obj_name = prompt_object['object']

        # 只生成数量为 10 的样本，其余数字一律跳过
        if required_object_num != 10:
            print(f"Skipping {obj_name} with {required_object_num} objects (only 10 is supported).")
            continue

        img_id = f'{obj_name}_num={required_object_num}_seed={seed}'
        vanilla_img = None

        if phase1_type == 'random_mask':
            object_masks = generate_random_masks_factory(shape=config['mask_creation']['random_mask']['shape'],
                                                         number_clusters=required_object_num)
            show_mask(object_masks, f"{config['pipeline']['output_path']}/{img_id}_mask.png")
            obj_num_match = False
        elif phase1_type == 'dbscan_mask':
            vanilla_masks, correct_number_mask, object_masks, vanilla_img, obj_num_match = spatial_prior_predictor(sdxl_pipe, prompt,
                                                                                                    required_object_num,
                                                                                                    config, seed)
            show_mask_list([vanilla_masks, correct_number_mask, object_masks],
                           titles=[f'Vanilla mask: {int(vanilla_masks.max())}',
                                   f'Correct number mask: {int(correct_number_mask.max())}',
                                   f'Postprocess mask: {int(object_masks.max())}'],
                           save_path=f"{config['pipeline']['output_path']}/{img_id}_masks.png")
        elif phase1_type == 'no_mask':
            object_masks = None

        if obj_num_match:
            image = vanilla_img
        else:
            if phase2_type == 'ours_counting_loss':
                # step_latent_noise_injection=0: 在管线开始前施加一次（默认，原行为）
                # step_latent_noise_injection>0: 在管线内部 step 0~N-1 逐步施加，此处跳过初始施加
                if step_latent_noise_injection == 0:
                    latents = apply_latent_partition(latents, object_masks)
                image = run_counting_pipeline_corrected_masks(
                    sdxl_pipe, prompt, generator, object_masks, latents, config,
                    step_latent_noise_injection=step_latent_noise_injection)
            elif phase2_type == 'vanilla':
                pass

        image.save(f"{out_dir}/{img_id}.png")
        vanilla_img.save(f"{out_dir}/{img_id}_vanilla.png")

        # Generate 3D cross-attention visualization (disabled to save memory)
        # visualize_and_save(sdxl_pipe, obj_name, required_object_num, seed, out_dir)

        # Generate per-step classification masks (spectral clustering) (disabled to save memory)
        # visualize_and_save_masks(sdxl_pipe, obj_name, required_object_num, seed, out_dir)

        metadata_item = {
            'id': img_id,
            'prompt': prompt,
            'seed': seed,
            'obj_class': obj_name,
            'requiered_object_num': required_object_num
        }

        metadata_json.append(metadata_item)

        with open(f"{out_dir}/metadata.json", "w") as f:
            json.dump(metadata_json, f, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--prompt", type=str, default="A photo of ten airplanes in the sky")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", type=str, default="pipeline/pipeline_config.yaml")
    parser.add_argument("--dataset_file", type=str, default="")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "--step_latent_noise_injection", type=int, default=0,
        help="潜空间分区植入步数。0=仅在管线开始前施加一次（默认）；"
             "N>0=在去噪 step 0~N-1 的每一步结束后施加（逐步累积）"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    config = read_yaml(args.config)

    if args.output_path is not None:
        config["pipeline"]["output_path"] = args.output_path

    if (args.dataset_file is not None) and (len(args.dataset_file) > 0):
        with open(args.dataset_file, "r") as f:
            all_objects = json.load(f)[:]
        # 只保留 int_number == 10 的样本
        prompt_objects = [obj for obj in all_objects if obj.get('int_number') == 10]
        print(f"Filtered dataset: {len(prompt_objects)} entries with int_number=10 "
              f"(out of {len(all_objects)} total)")

    else:
        int_number, object_singular = find_nummod([args.prompt])[0]
        num = word2number.get(int_number, 0)
        if num != 10:
            print(f"Error: This script only supports count=10, got {num}. Exiting.")
            sys.exit(1)
        prompt_objects = [{
            "prompt": args.prompt,
            "object": object_singular,
            "int_number": num,
            "seed": args.seed
        }, ]

    phase1_type = config['pipeline']['phase1_type']
    phase2_type = config['pipeline']['phase2_type']

    run_pipeline(prompt_objects, config, phase1_type, phase2_type,
                 step_latent_noise_injection=args.step_latent_noise_injection)
