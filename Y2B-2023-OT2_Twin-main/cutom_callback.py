import os
import wandb
from wandb.integration.sb3 import WandbCallback
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

class CustomWandbCallback(WandbCallback):
    """Custom callback to save the model periodically without overwriting it, and upload it to Weights and Biases."""

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
    ) -> None:
        super().__init__(verbose, model_save_path, model_save_freq, gradient_save_freq, log)

    def _init_callback(self) -> None:
        """Initialize the callback, setting hyperparameters and logging setup."""
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        if self.gradient_save_freq > 0:
            wandb.watch(
                self.model.policy,
                log_freq=self.gradient_save_freq,
                log=self.log,
            )
        wandb.config.setdefaults(d)

    def _on_step(self) -> bool:
        """Called after each step in training to check if the model should be saved."""
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.n_calls % self.model_save_freq == 0:
                    self.save_model()
        return True

    def _on_training_end(self) -> None:
        """Called at the end of training to ensure the model is saved one last time."""
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        """Override the save_model method to save the model with a unique name at each save."""
        # Generate a unique filename based on the current timestamp and training steps (n_calls)
        model_save_path = os.path.join(self.model_save_path, f"model_{self.n_calls}.zip")
        
        # Save the model
        self.model.save(model_save_path)
        
        # Upload model to wandb
        wandb.save(model_save_path, base_path=self.model_save_path)
        
        # Log the model save action
        if self.verbose > 1:
            logger.info(f"Saving model checkpoint to {model_save_path}")
