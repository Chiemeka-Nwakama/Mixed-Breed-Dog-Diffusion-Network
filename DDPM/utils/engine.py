# utils/engine.py
# Core diffusion utilities: training, DDPM sampler, DDIM mixed sampler, and mixed DDPM sampler.

from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def extract(v: torch.Tensor, i: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Gather v[i] with broadcasting to match 'shape'.
    v: typically shape (T,), i: (batch,)
    Returns: (batch, 1, 1, 1, ...)
    """
    out = torch.gather(v, index=i, dim=0).to(device=i.device, dtype=torch.float32)
    return out.view([i.shape[0]] + [1] * (len(shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    """
    Training forward pass: add noise at random t and predict it with the model.
    """
    def __init__(self, model: nn.Module, beta: Tuple[float, float], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of α , named ᾱ_t in paper
        alpha_t = 1.0 - self.beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of q(x_t | x_0)
        self.register_buffer("signal_rate", torch.sqrt(alpha_bar_t))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_bar_t))

    def forward(self, x_0: torch.Tensor, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Training forward pass - adds noise and predicts it
        """
        # Get a random training step t ~ Uniform({1, ..., T})
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # Generate noise ε ~ N(0, 1)
        epsilon = torch.randn_like(x_0)

        # Create noisy image x_t using the noise schedule
        x_t = extract(self.signal_rate, t, x_0.shape) * x_0 + extract(self.noise_rate, t, x_0.shape) * epsilon
        
        # Predict the noise using the model WITH class conditioning
        epsilon_theta = self.model(x_t, t, class_labels)

        # Calculate loss (MSE between predicted and actual noise)
        loss = F.mse_loss(epsilon_theta, epsilon, reduction="mean")
        return loss
    


class DDPMSampler(nn.Module):
    """
    Standard DDPM sampler with classifier-free guidance and optional mixed conditioning.
    - Supports class_labels = None (uncond), Tensor of indices, or (labels_1, labels_2, mix_ratio) tuple.
    """
    def __init__(self, model: nn.Module, beta: Tuple[float, float], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of α , named ᾱ_t in paper
        alpha_t = 1.0 - self.beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        alpha_bar_prev = F.pad(alpha_bar_t[:-1], (1, 0), value=1.0)

        self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))
        self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t))
        self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))

    @torch.no_grad()
    def _posterior_mean_var(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor):
        mean = extract(self.coeff_1, t, x_t.shape) * x_t - extract(self.coeff_2, t, x_t.shape) * eps_pred
        var = extract(self.posterior_variance, t, x_t.shape)
        return mean, var

    @torch.no_grad()
    def forward(
        self,
        x_T: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        only_return_x_0: bool = True,
        interval: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Parameters:
            x_T: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            only_return_x_0: Determines whether to return only x_0.
            interval: Saving interval when returning trajectory.
            kwargs: for compatibility.
        """
        x_t = x_T
        traj = [x_t]

        with tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T) as sampling_steps:
            for time_step in sampling_steps:

                # time label
                t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
                # Classifier-Free Guidance Noise Prediction
                if guidance_scale > 1.0 and class_labels is not None:
                    # Support mixed conditioning: (labels_1, labels_2, mix_ratio)
                    if isinstance(class_labels, tuple) and len(class_labels) == 3:
                        labels_1, labels_2, mix_ratio = class_labels
                        eps_uncond = self.model(x_t, t, None)
                        eps_1 = self.model(x_t, t, labels_1)
                        eps_2 = self.model(x_t, t, labels_2)
                        cond_mix = mix_ratio * (eps_1 - eps_uncond) + (1.0 - mix_ratio) * (eps_2 - eps_uncond)
                        eps_pred = eps_uncond + guidance_scale * cond_mix
                    else:
                        # Conditional + unconditional prediction
                        eps_cond = self.model(x_t, t, class_labels)
                        eps_uncond = self.model(x_t, t, None)
                        # Classifier-free guidance
                        eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    eps_pred = self.model(x_t, t, class_labels)

                mean, var = self._posterior_mean_var(x_t, t, eps_pred)

                z = torch.randn_like(x_t) if time_step > 0 else 0
                x_t = mean + torch.sqrt(var) * z

                if not only_return_x_0 and ((self.T - time_step) % interval == 0 or time_step == 0):
                    traj.append(torch.clamp(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": time_step + 1, "sample": len(traj)})

        return x_t if only_return_x_0 else torch.stack(traj, dim=1)


class GradualMixedBreedSampler(nn.Module):
    """
    Alternative approach: Gradually transition from separate to mixed.
    Smoother blending with adjustable transition curve.
    """
    def __init__(self, model: nn.Module, beta: Tuple[float, float], T: int):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
        alpha_t = 1.0 - self.beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        alpha_bar_prev = F.pad(alpha_bar_t[:-1], (1, 0), value=1.0)

        self.register_buffer("alpha_bar_t", alpha_bar_t)
        self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))
        self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t))
        self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))

    def get_blend_weight(self, time_step: int, strategy: str = "linear") -> float:
        

        """
        Returns how much to blend at this timestep.
        0.0 = completely separate breeds
        1.0 = fully mixed

        Strategies:
        - linear: Smooth linear transition
        - sigmoid: S-curve (slow-fast-slow)
        - late: Stay separate longer, mix at end
        """
        progress = 1.0 - (time_step / self.T)  # float

        if strategy == "linear":
            return progress

        elif strategy == "sigmoid":
            # Convert float → Tensor before exp
            x = torch.tensor(progress, dtype=torch.float32, device=self.beta_t.device)
            return (1.0 / (1.0 + torch.exp(-10 * (x - 0.5)))).item()

        elif strategy == "late":
            return max(0.0, (progress - 0.6) / 0.4)

        return progress



    @torch.no_grad()
    def _posterior_mean_var(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor):
        mean = extract(self.coeff_1, t, x_t.shape) * x_t - extract(self.coeff_2, t, x_t.shape) * eps_pred
        var = extract(self.posterior_variance, t, x_t.shape)
        return mean, var

    @torch.no_grad()
    def forward(
        self,
        z_A: torch.Tensor,
        z_B: torch.Tensor,
        labels_A: torch.Tensor,
        labels_B: torch.Tensor,
        mix_ratio: float = 0.5,
        blend_strategy: str = "sigmoid",
        guidance_scale: float = 5.0,
        use_cfg: bool = True,
        only_return_x_0: bool = True,
        interval: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """Gradually blend from separate to mixed"""
        x_A = z_A
        x_B = z_B
        traj = []

        with tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T) as steps:
            for time_step in steps:
                t = torch.full((z_A.shape[0],), time_step, device=z_A.device, dtype=torch.long)
                
                # Get blend weight for this timestep
                blend_weight = self.get_blend_weight(time_step, blend_strategy)
                
                # Get predictions
                if use_cfg and guidance_scale > 1.0:
                    try:
                        eps_uncond_A = self.model(x_A, t, None)
                        eps_cond_A = self.model(x_A, t, labels_A)
                        eps_A = eps_uncond_A + guidance_scale * (eps_cond_A - eps_uncond_A)
                        
                        eps_uncond_B = self.model(x_B, t, None)
                        eps_cond_B = self.model(x_B, t, labels_B)
                        eps_B = eps_uncond_B + guidance_scale * (eps_cond_B - eps_uncond_B)
                    except:
                        eps_A = self.model(x_A, t, labels_A)
                        eps_B = self.model(x_B, t, labels_B)
                        use_cfg = False
                else:
                    eps_A = self.model(x_A, t, labels_A)
                    eps_B = self.model(x_B, t, labels_B)
                
                # Blend the predictions based on current blend weight
                # blend_weight=0: keep separate, blend_weight=1: fully mix
                if blend_weight < 0.01:
                    # Completely separate
                    mean_A, var = self._posterior_mean_var(x_A, t, eps_A)
                    mean_B, _ = self._posterior_mean_var(x_B, t, eps_B)
                    z = torch.randn_like(x_A) if time_step > 0 else 0
                    x_A = mean_A + torch.sqrt(var) * z
                    x_B = mean_B + torch.sqrt(var) * z
                    x_result = mix_ratio * x_A + (1.0 - mix_ratio) * x_B
                else:
                    # Blend the latents and predictions
                    x_blended = blend_weight * (mix_ratio * x_A + (1.0 - mix_ratio) * x_B) + \
                               (1.0 - blend_weight) * x_A  # Weighted average
                    
                    eps_blended = blend_weight * (mix_ratio * eps_A + (1.0 - mix_ratio) * eps_B) + \
                                 (1.0 - blend_weight) * eps_A
                    
                    mean, var = self._posterior_mean_var(x_blended, t, eps_blended)
                    z = torch.randn_like(x_blended) if time_step > 0 else 0
                    x_result = mean + torch.sqrt(var) * z
                    
                    # Update both for next iteration
                    x_A = x_result
                    x_B = x_result
                
                if not only_return_x_0 and ((self.T - time_step) % interval == 0 or time_step == 0):
                    traj.append(torch.clamp(x_result, -1.0, 1.0))
                
                steps.set_postfix(step=time_step + 1, blend=f"{blend_weight:.2f}")

        return x_result if only_return_x_0 else torch.stack(traj, dim=1)