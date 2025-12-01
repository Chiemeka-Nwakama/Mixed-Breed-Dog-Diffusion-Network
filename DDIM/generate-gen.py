# generate.py - Generates images from a trained DDPM model
from argparse import ArgumentParser
import torch
from model.UNet import UNet
from utils.engine import DDPMSampler
from utils.tools import save_image

def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm"])

    # Conditional generation
    parser.add_argument("--class_1", type=int, help="Breed class (0-11)")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="Classifier-free guidance scale (2-5 recommended)")

    # Generator parameters
    parser.add_argument("-bs", "--batch_size", type=int, default=4)
    parser.add_argument("--unconditional", action="store_true", help="Generate without class labels")

    # Sampler parameters
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])

    # Save image parameters
    parser.add_argument("--nrow", type=int, default=2)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)

    return parser.parse_args()


@torch.no_grad()
def generate_single(args):
    device = torch.device(args.device)

    # Load checkpoint
    cp = torch.load(args.checkpoint_path, map_location=device)

    # Load model
    model = UNet(**cp["config"]["Model"])
    model.load_state_dict(cp["model"])
    model.to(device)
    model.eval()

    # Create sampler
    if args.sampler == "ddpm":
        sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    # Handle image_size: int -> tuple
    image_size = cp["config"]["Dataset"]["image_size"]
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    # Generate Gaussian noise
    z_t = torch.randn((args.batch_size, cp["config"]["Model"]["in_channels"], *image_size),
                      device=device)

    # Create class labels if not unconditional
    if args.unconditional:
        labels = None
        print("Generating images unconditionally (no class labels)...")
    else:
        if args.class_1 is None:
            raise ValueError("For conditional generation, --class_1 must be specified")
        labels = torch.full((args.batch_size,), args.class_1, dtype=torch.long, device=device)
        print(f"Generating breed class {args.class_1}...")
        print(f"Guidance scale: {args.guidance_scale}")

    print(f"Sampler: {args.sampler.upper()}")
    extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)

    # Generate images
    x = sampler(z_t, class_labels=labels, guidance_scale=args.guidance_scale,
                only_return_x_0=True, **extra_param)

    # Save image
    save_path = args.image_save_path or (f"class_{args.class_1}.png" if labels is not None else "unconditional.png")
    save_image(x, nrow=args.nrow, show=args.show, path=save_path)

    print(f"Saved generated images to: {save_path}")


if __name__ == "__main__":
    args = parse_option()
    generate_single(args)
