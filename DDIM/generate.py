from utils.engine import DDPMSampler, DDIMSampler
from model.UNet import UNet
import torch
from utils.tools import save_image
from argparse import ArgumentParser


def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])

    #  Mixed breed parameters
    parser.add_argument("--class_1", type=int, required=True, help="First breed class (0-11)")
    parser.add_argument("--class_2", type=int, required=True, help="Second breed class (0-11)")
    parser.add_argument("--mix_ratio", type=float, default=0.5, 
                       help="Mix ratio: 0.5=50/50, 0.7=70%% class_1 + 30%% class_2")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                       help="Classifier-free guidance scale (2-5 recommended)")
  

    # generator param
    parser.add_argument("-bs", "--batch_size", type=int, default=4)

    # DDIM sampler param (recommended for mixed generation)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])

    # save image param
    parser.add_argument("--nrow", type=int, default=2)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate_mixed(args):
    device = torch.device(args.device)

    cp = torch.load(args.checkpoint_path)
    
    # Load trained model
    model = UNet(**cp["config"]["Model"])
    model.load_state_dict(cp["model"])
    model.to(device)
    model.eval()

    # Create samplers for both classes
    if args.sampler == "ddim":
        sampler = DDIMSampler(model, **cp["config"]["Trainer"]).to(device)
    elif args.sampler == "ddpm":
        sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    # Generate Gaussian noise (same starting point for both)
    z_t = torch.randn((args.batch_size, cp["config"]["Model"]["in_channels"],
                       *cp["config"]["Dataset"]["image_size"]), device=device)

    # Create class labels
    labels_1 = torch.full((args.batch_size,), args.class_1, dtype=torch.long, device=device)
    labels_2 = torch.full((args.batch_size,), args.class_2, dtype=torch.long, device=device)

    print(f"Generating mixed breed:")
    print(f"  {args.mix_ratio*100:.0f}% Class {args.class_1} + {(1-args.mix_ratio)*100:.0f}% Class {args.class_2}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Sampler: {args.sampler.upper()}")

    # Generating breed with mixed conditioning
    extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)
    
    # Generate with class 1
    x_1 = sampler(z_t.clone(), class_labels=labels_1, guidance_scale=args.guidance_scale,
                  only_return_x_0=True, **extra_param)
    
    # Generate with class 2 (using same noise!)
    x_2 = sampler(z_t.clone(), class_labels=labels_2, guidance_scale=args.guidance_scale,
                  only_return_x_0=True, **extra_param)
    
    # Mix the results
    x_mixed = args.mix_ratio * x_1 + (1 - args.mix_ratio) * x_2
   

    # Save result
    save_path = args.image_save_path or f"mixed_class{args.class_1}_class{args.class_2}_ratio{args.mix_ratio}.png"
    save_image(x_mixed, nrow=args.nrow, show=args.show, path=save_path)
    
    print(f"Saved mixed breed images to: {save_path}")


if __name__ == "__main__":
    args = parse_option()
    generate_mixed(args)