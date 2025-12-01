from utils.engine import LatentMixedBreedSampler, GradualMixedBreedSampler
from model.UNet import UNet
import torch
from utils.tools import save_image
from argparse import ArgumentParser

def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Breed selection
    parser.add_argument("--class_1", type=int, required=True)
    parser.add_argument("--class_2", type=int, required=True)
    parser.add_argument("--mix_ratio", type=float, default=0.5)
    
    # Sampler choice
    parser.add_argument("--sampler", type=str, default="latent",
                       choices=["latent", "gradual"],
                       help="latent: 2-stage mix in latent space | gradual: smooth transition")
    
    # Latent sampler settings
    parser.add_argument("--mix_timestep", type=int, default=None,
                       help="When to mix breeds (default: T//2). Lower=later mix, higher=earlier")
    
    # Gradual sampler settings  
    parser.add_argument("--blend_strategy", type=str, default="sigmoid",
                       choices=["linear", "sigmoid", "late"],
                       help="How to transition from separate to mixed")
    
    # Quality settings
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--use_dynamic_threshold", action="store_true")
    
    # Generation
    parser.add_argument("-bs", "--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    
    # Save
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)
    
    args = parser.parse_args()
    return args

@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Load model
    print("Loading model...")
    cp = torch.load(args.checkpoint_path, map_location=device)
    model = UNet(**cp["config"]["Model"])
    model.load_state_dict(cp["model"])
    model.to(device).eval()
    
    # Validate
    num_classes = cp["config"]["Model"].get("num_classes", 120)
    if not (0 <= args.class_1 < num_classes and 0 <= args.class_2 < num_classes):
        raise ValueError(f"Classes must be in range [0, {num_classes-1}]")
    
    # Choose sampler
    if args.sampler == "latent":
        print("Using Latent-Space Mixed Sampler (2-stage approach)")
        sampler = LatentMixedBreedSampler(model, **cp["config"]["Trainer"]).to(device)
    else:
        print("Using Gradual Mixed Sampler (smooth transition)")
        sampler = GradualMixedBreedSampler(model, **cp["config"]["Trainer"]).to(device)
    
    # Prepare
    image_size = cp["config"]["Dataset"]["image_size"]
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    
    z_A = torch.randn((args.batch_size, cp["config"]["Model"]["in_channels"], *image_size), device=device)
    z_B = torch.randn_like(z_A)
    
    labels_A = torch.full((args.batch_size,), args.class_1, dtype=torch.long, device=device)
    labels_B = torch.full((args.batch_size,), args.class_2, dtype=torch.long, device=device)
    
    print(f"\n{'='*70}")
    print(f"Mixed Breed Generation:")
    print(f"  Breeds: {args.mix_ratio*100:.0f}% Class {args.class_1} + "
          f"{(1-args.mix_ratio)*100:.0f}% Class {args.class_2}")
    print(f"  Sampler: {args.sampler}")
    
    if args.sampler == "latent":
        mix_step = args.mix_timestep or cp["config"]["Trainer"]["T"] // 2
        print(f"  Mix timestep: {mix_step} (out of {cp['config']['Trainer']['T']})")
        print(f"    → Lower = mix later (more distinct features)")
        print(f"    → Higher = mix earlier (more blended)")
    else:
        print(f"  Blend strategy: {args.blend_strategy}")
    
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Dynamic threshold: {args.use_dynamic_threshold}")
    print(f"{'='*70}\n")
    
    # Generate!
    if args.sampler == "latent":
        x_mixed = sampler(
            z_A, z_B, labels_A, labels_B,
            mix_ratio=args.mix_ratio,
            mix_timestep=args.mix_timestep,
            guidance_scale=args.guidance_scale,
            use_cfg=args.use_cfg,
            use_dynamic_threshold=args.use_dynamic_threshold,
            only_return_x_0=True
        )
    else:
        x_mixed = sampler(
            z_A, z_B, labels_A, labels_B,
            mix_ratio=args.mix_ratio,
            blend_strategy=args.blend_strategy,
            guidance_scale=args.guidance_scale,
            use_cfg=args.use_cfg,
            only_return_x_0=True
        )
    
    # Save
    if args.image_save_path:
        save_path = args.image_save_path
    else:
        save_path = f"mixed_{args.sampler}_c{args.class_1}_c{args.class_2}_r{args.mix_ratio}_cfg{args.guidance_scale}.png"
    
    save_image(x_mixed, nrow=args.nrow, show=args.show, path=save_path)
    print(f"\n Saved to: {save_path}")
    
    # Test same-breed mixing
    if args.class_1 == args.class_2:
        print("\n  Note: You're mixing the same breed with itself.")
        print("    Don’t expect improved results — the output may look the same or sometimes worse.")
        print("     If the images degrade, that’s normal for this sampler.")

if __name__ == "__main__":
    args = parse_option()
    main(args)