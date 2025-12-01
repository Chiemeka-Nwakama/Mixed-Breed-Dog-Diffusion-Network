from utils.engine import DDPMSampler
from model.UNet import UNet
import torch
from utils.tools import save_image
from argparse import ArgumentParser


def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm"])

    # Single breed parameters
    parser.add_argument("--class_1", type=int, required=True, help="Breed class (0-11)")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="Classifier-free guidance scale (2-5 recommended)")

    # Generator parameters
    parser.add_argument("-bs", "--batch_size", type=int, default=4)

    # Sampler parameters
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])

    # Save image parameters
    parser.add_argument("--nrow", type=int, default=2)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate_single(args):
    device = torch.device(args.device)

    cp = torch.load(args.checkpoint_path)

    # Load trained model
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
    # Create class labels
    labels = torch.full((args.batch_size,), args.class_1, dtype=torch.long, device=device)

    print(f"Generating breed class {args.class_1}...")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Sampler: {args.sampler.upper()}")

    # Generate images
    extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)
    x = sampler(z_t, class_labels=labels, guidance_scale=args.guidance_scale,
                only_return_x_0=True, **extra_param)

    # Save result
    save_path = args.image_save_path or f"class_{args.class_1}.png"
    save_image(x, nrow=args.nrow, show=args.show, path=save_path)

    print(f"Saved generated images to: {save_path}")


if __name__ == "__main__":
    args = parse_option()
    generate_single(args)
