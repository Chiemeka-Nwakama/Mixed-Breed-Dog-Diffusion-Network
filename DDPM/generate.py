# generate.py - generator for unconditional, single-class, and mixed-breed  dog diffusion models

from argparse import ArgumentParser
import torch
from model.UNet import UNet
from utils.engine import DDPMSampler, GradualMixedBreedSampler
from utils.tools import save_image



# Argument Parser

def parse_option():
    parser = ArgumentParser()

    # Required checkpoint
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Mode flags
    parser.add_argument("--unconditional", action="store_true",
                        help="Unconditional generation: no labels")

    parser.add_argument("--single", action="store_true",
                        help="Single-breed conditional generation")

    parser.add_argument("--mixed", action="store_true",
                        help="Mixed-breed generation")

    # Class inputs
    parser.add_argument("--class_1", type=int, help="Primary class (0-N) for single or mixed")
    parser.add_argument("--class_2", type=int, help="Secondary class (for mixed)")
    parser.add_argument("--mix_ratio", type=float, default=0.5,
                        help="For mixed mode: fraction of class_1")

    # Guidance
    parser.add_argument("--guidance_scale", type=float, default=3.0)

    # Batch size
    parser.add_argument("-bs", "--batch_size", type=int, default=4)

    # Sampler settings
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="linear",
                        choices=["linear", "quadratic"])

    # Mixed-breed sampler options
    parser.add_argument("--sampler_type", type=str, default="gradual",
                        choices=["gradual", "latent"])

    parser.add_argument("--mix_timestep", type=int, default=None)
    parser.add_argument("--blend_strategy", type=str, default="sigmoid",
                        choices=["linear", "sigmoid", "late"])

    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--use_dynamic_threshold", action="store_true")

    # Save options
    parser.add_argument("--nrow", type=int, default=2)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)

    return parser.parse_args()



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
