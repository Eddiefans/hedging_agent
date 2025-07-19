from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CurriculumCallback(BaseCallback):
    def __init__(self, 
                 total_timesteps: int,
                 stage_proportions=[0.1, 0.1, 0.2, 0.6],  # Stage proportions 
                 stage_names=["bull", "bear", "mixed", "random"],   # Stage names
                 verbose=1):
        super().__init__(verbose)
        
        # Validate proportions
        if abs(sum(stage_proportions) - 1.0) > 1e-6:
            raise ValueError(f"Stage proportions must sum to 1.0, got {sum(stage_proportions)}")
        
        self.total_timesteps = total_timesteps
        self.stage_proportions = stage_proportions
        self.stage_names = stage_names
        self.current_stage = 0
        
        # Calculate actual timesteps for each stage
        self.stage_timesteps = [int(total_timesteps * prop) for prop in stage_proportions]
        
        # Adjust for rounding errors 
        remainder = total_timesteps - sum(self.stage_timesteps)
        self.stage_timesteps[-1] += remainder
        
        self.cumulative_timesteps = np.cumsum([0] + self.stage_timesteps)
        
        if verbose >= 1:
            print(f"\n=== CURRICULUM LEARNING SCHEDULE ===")
            for i, (name, timesteps, prop) in enumerate(zip(stage_names, self.stage_timesteps, stage_proportions)):
                start_ts = self.cumulative_timesteps[i]
                end_ts = self.cumulative_timesteps[i + 1]
                print(f"Stage {i+1}: {name:<6} | {timesteps:>7,} steps ({prop*100:4.1f}%) | Timesteps {start_ts:>7,}-{end_ts:>7,}")
            print(f"=====================================\n")
        
        
    def _on_step(self) -> bool:
        # Check if we should advance to the next curriculum stage
        current_timestep = self.num_timesteps
        
        # Find which stage we should be in
        new_stage = 0
        for i, threshold in enumerate(self.cumulative_timesteps[1:]):
            if current_timestep < threshold:
                new_stage = i
                break
        else:
            new_stage = len(self.stage_names) - 1  # Final stage
        
        # If stage changed, update environment
        if new_stage != self.current_stage:
            self.current_stage = new_stage
            stage_name = self.stage_names[new_stage]
            
            if self.verbose >= 1:
                print(f"\n=== CURRICULUM STAGE CHANGE ===")
                print(f"Timestep: {current_timestep}")
                print(f"New Stage: {stage_name} (Stage {new_stage + 1}/{len(self.stage_names)})")
                print(f"================================\n")
            
            # Update training environment curriculum stage
            if hasattr(self.training_env, 'set_curriculum_stage'):
                self.training_env.set_curriculum_stage(stage_name)
            elif hasattr(self.training_env, 'envs'):  # VecEnv case
                for env in self.training_env.envs:
                    if hasattr(env, 'set_curriculum_stage'):
                        env.set_curriculum_stage(stage_name)
        
        return True