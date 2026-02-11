# generate.py - generator for unconditional, single-class, and mixed-breed  dog diffusion models

from argparse import Namespace
import torch
from model.UNet import UNet
from utils.engine import DDPMSampler, GradualMixedBreedSampler
from utils.tools import save_image

def input_bool(prompt):
    val = input(prompt + " (y/n): ").strip().lower()
    return val in ["y", "yes"]

def parse_option():
    print("Dog Diffusion Model Generator")
    print("Select generation mode:")
    print("1. Unconditional")
    print("2. Single-breed conditional")
    print("3. Mixed-breed")
    mode = input("Enter 1, 2, or 3: ").strip()
    unconditional = mode == "1"
    single = mode == "2"
    mixed = mode == "3"

    checkpoint_path = input("Enter checkpoint path: ").strip()
    device = input("Device [cuda]: ").strip() or "cuda"

    class_1 = None
    class_2 = None
    mix_ratio = 0.5

    if single or mixed:
        class_1 = int(input("Primary class (0-N): ").strip())
    if mixed:
        class_2 = int(input("Secondary class (for mixed): ").strip())
        mix_ratio = float(input("Mix ratio (fraction of class_1, 0-1) [0.5]: ").strip() or 0.5)

    guidance_scale = float(input("Guidance scale [3.0]: ").strip() or 3.0)
    batch_size = int(input("Batch size [4]: ").strip() or 4)
    eta = float(input("Eta [0.0]: ").strip() or 0.0)
    steps = int(input("Steps [100]: ").strip() or 100)
    method = input("Method [linear/quadratic]: ").strip() or "linear"

    sampler_type = "gradual"
    mix_timestep = None
    blend_strategy = "sigmoid"
    if mixed:
        sampler_type = input("Sampler type [gradual/latent]: ").strip() or "gradual"
        if sampler_type == "latent":
            mix_timestep_in = input("Mix timestep (int or blank): ").strip()
            mix_timestep = int(mix_timestep_in) if mix_timestep_in else None
        else:
            blend_strategy = input("Blend strategy [linear/sigmoid/late]: ").strip() or "sigmoid"

    use_cfg = input_bool("Use classifier-free guidance?")
    use_dynamic_threshold = input_bool("Use dynamic thresholding?")

    nrow = int(input("nrow for image grid [2]: ").strip() or 2)
    show = input_bool("Show image after generation?")
    image_save_path = input("Image save path (blank for default): ").strip() or None

    return Namespace(
        checkpoint_path=checkpoint_path,
        device=device,
        unconditional=unconditional,
        single=single,
        mixed=mixed,
        class_1=class_1,
        class_2=class_2,
        mix_ratio=mix_ratio,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        eta=eta,
        steps=steps,
        method=method,
        sampler_type=sampler_type,
        mix_timestep=mix_timestep,
        blend_strategy=blend_strategy,
        use_cfg=use_cfg,
        use_dynamic_threshold=use_dynamic_threshold,
        nrow=nrow,
        show=show,
        image_save_path=image_save_path
    )

# Load model + return config

def load_model_and_sampler(args):
    device = torch.device(args.device)

    cp = torch.load(args.checkpoint_path, map_location=device)
    config = cp["config"]

    model = UNet(**config["Model"])
    model.load_state_dict(cp["model"])
    model.to(device).eval()

    # Standard DDPM sampler (always needed)
    ddpm = DDPMSampler(model, **config["Trainer"]).to(device)

    return model, ddpm, cp, config, device


# Unconditional generation

@torch.no_grad()
def run_unconditional(args):
    model, sampler, cp, config, device = load_model_and_sampler(args)

    print("Unconditional generation: no class labels used.")

    # Handle image size
    image_size = config["Dataset"]["image_size"]
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    # Noise
    z_t = torch.randn(
        (args.batch_size, config["Model"]["in_channels"], *image_size),
        device=device
    )

    # Generate
    x = sampler(
        z_t,
        class_labels=None,
        guidance_scale=args.guidance_scale,
        only_return_x_0=True,
        steps=args.steps,
        eta=args.eta,
        method=args.method
    )

    save_path = args.image_save_path or "unconditional.png"
    save_image(x, nrow=args.nrow, show=args.show, path=save_path)
    print("Saved:", save_path)



# Single-class conditional generation

@torch.no_grad()
def run_single(args):
    if args.class_1 is None:
        raise ValueError("single mode requires --class_1")

    model, sampler, cp, config, device = load_model_and_sampler(args)

    print(f"Single conditional generation for class {args.class_1}")

    image_size = config["Dataset"]["image_size"]
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    z_t = torch.randn(
        (args.batch_size, config["Model"]["in_channels"], *image_size),
        device=device
    )

    labels = torch.full((args.batch_size,), args.class_1, dtype=torch.long, device=device)

    x = sampler(
        z_t,
        class_labels=labels,
        guidance_scale=args.guidance_scale,
        only_return_x_0=True,
        steps=args.steps,
        eta=args.eta,
        method=args.method
    )

    save_path = args.image_save_path or f"class_{args.class_1}.png"
    save_image(x, nrow=args.nrow, show=args.show, path=save_path)
    print("Saved:", save_path)


# ------------------------------
# Mixed breed generation
# ------------------------------
@torch.no_grad()
def run_mixed(args):
    if args.class_1 is None or args.class_2 is None:
        raise ValueError("mixed mode requires --class_1 and --class_2")

    model, ddpm_sampler, cp, config, device = load_model_and_sampler(args)

    num_classes = config["Model"].get("num_classes", 120)
    if not (0 <= args.class_1 < num_classes and 0 <= args.class_2 < num_classes):
        raise ValueError("class index out of range")

    # Choose sampler type
    if args.sampler_type == "latent":
        sampler = LatentMixedBreedSampler(model, **config["Trainer"]).to(device)
    else:
        sampler = GradualMixedBreedSampler(model, **config["Trainer"]).to(device)

    image_size = config["Dataset"]["image_size"]
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    z_A = torch.randn((args.batch_size, config["Model"]["in_channels"], *image_size), device=device)
    z_B = torch.randn_like(z_A)

    labels_A = torch.full((args.batch_size,), args.class_1, dtype=torch.long, device=device)
    labels_B = torch.full((args.batch_size,), args.class_2, dtype=torch.long, device=device)

    print("Mixed-breed generation:")
    print(f"  {args.mix_ratio*100:.0f}% class {args.class_1} + {100 - args.mix_ratio*100:.0f}% class {args.class_2}")

    if args.sampler_type == "latent":
        x = sampler(
            z_A, z_B, labels_A, labels_B,
            mix_ratio=args.mix_ratio,
            mix_timestep=args.mix_timestep,
            guidance_scale=args.guidance_scale,
            use_cfg=args.use_cfg,
            use_dynamic_threshold=args.use_dynamic_threshold,
            only_return_x_0=True
        )
    else:
        x = sampler(
            z_A, z_B, labels_A, labels_B,
            mix_ratio=args.mix_ratio,
            blend_strategy=args.blend_strategy,
            guidance_scale=args.guidance_scale,
            use_cfg=args.use_cfg,
            only_return_x_0=True
        )

    save_path = args.image_save_path or f"mixed_c{args.class_1}_c{args.class_2}_{args.mix_ratio}.png"
    save_image(x, nrow=args.nrow, show=args.show, path=save_path)

    print("Saved:", save_path)



# Main

if __name__ == "__main__":
    args = parse_option()

    if args.unconditional:
        run_unconditional(args)

    elif args.single:
        run_single(args)

    elif args.mixed:
        run_mixed(args)

    else:
        raise ValueError("Must select one of: --unconditional, --single, or --mixed")