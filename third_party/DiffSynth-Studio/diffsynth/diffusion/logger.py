import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self._optimizer = None
        self._scheduler = None
        self._accelerator = None
        # [NEW] for dataloader resume - the runner refreshes this just before every step
        self._current_epoch = 0
        self._current_step_in_epoch = 0


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        file_name = f"epoch-{epoch_id}.safetensors"
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
        # [NEW] save per-epoch too, through accelerator.save_state, so optimizer and scheduler go with it
        if self._accelerator is not None:
            accel_state_dir = os.path.join(self.output_path, file_name.replace('.safetensors', '_accel_state'))
            self._accelerator.save_state(accel_state_dir)
            if accelerator.is_main_process:
                import json as _json
                with open(os.path.join(accel_state_dir, 'custom_state.json'), 'w') as _f:
                    _json.dump({
                        'num_steps': self.num_steps,
                        'epoch_id': self._current_epoch,
                        'step_in_epoch': self._current_step_in_epoch,
                    }, _f)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
        # [NEW] accelerator.save_state writes optimizer, scheduler and model together
        if self._accelerator is not None:
            accel_state_dir = os.path.join(self.output_path, file_name.replace('.safetensors', '_accel_state'))
            self._accelerator.save_state(accel_state_dir)
            # keep num_steps with it
            if accelerator.is_main_process:
                import json as _json
                with open(os.path.join(accel_state_dir, 'custom_state.json'), 'w') as _f:
                    _json.dump({
                        'num_steps': self.num_steps,
                        'epoch_id': self._current_epoch,
                        'step_in_epoch': self._current_step_in_epoch,
                    }, _f)
