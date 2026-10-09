import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    # [Modified] T5/CLIP을 CPU에 남겨두고 나머지만 GPU로 이동
    # 14B DiT + T5 + CLIP이 한 번에 GPU에 올라가면 80GB를 초과하므로
    # T5/CLIP은 forward 시 필요할 때만 GPU에 올림 (GRACEWanTrainingModule.forward)
    _pipe = getattr(model, 'pipe', None)
    _t5_hold = _clip_hold = None
    if _pipe is not None:
        _t5_hold = getattr(_pipe, 'text_encoder', None)
        _clip_hold = getattr(_pipe, 'image_encoder', None)
        if _t5_hold is not None: _pipe.text_encoder = None
        if _clip_hold is not None: _pipe.image_encoder = None
    model.to(device=accelerator.device)
    if _pipe is not None:
        if _t5_hold is not None: _pipe.text_encoder = _t5_hold    # stays on CPU
        if _clip_hold is not None: _pipe.image_encoder = _clip_hold  # stays on CPU
    torch.cuda.empty_cache()
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    # [NEW] optimizer/scheduler/accelerator를 logger에 등록 (checkpoint 저장/로드용)
    model_logger._optimizer = optimizer
    model_logger._scheduler = scheduler
    model_logger._accelerator = accelerator
    # [NEW] resume 진행 위치
    _resume_epoch = 0
    _resume_step_in_epoch = 0
    # [NEW] accelerator state resume
    if hasattr(args, 'lora_checkpoint') and args.lora_checkpoint:
        _accel_state_dir = args.lora_checkpoint.replace('.safetensors', '_accel_state')
        if os.path.exists(_accel_state_dir):
            accelerator.load_state(_accel_state_dir)
            # [NEW] num_steps + dataloader 위치 복원
            _custom_path = os.path.join(_accel_state_dir, 'custom_state.json')
            if os.path.exists(_custom_path):
                import json as _json
                with open(_custom_path) as _f:
                    _custom = _json.load(_f)
                model_logger.num_steps = int(_custom.get('num_steps', 0))
                _resume_epoch = int(_custom.get('epoch_id', 0))
                _resume_step_in_epoch = int(_custom.get('step_in_epoch', 0))
            if accelerator.is_main_process:
                print(f"[resume] Accelerator state loaded from {_accel_state_dir} "
                      f"(num_steps={model_logger.num_steps}, epoch={_resume_epoch}, step_in_epoch={_resume_step_in_epoch})")
        else:
            # fallback: 이전 형식 (_optim.pt)
            _optim_path = args.lora_checkpoint.replace('.safetensors', '_optim.pt')
            if os.path.exists(_optim_path):
                if accelerator.is_main_process:
                    print(f"[resume] WARNING: Using legacy _optim.pt (may not work correctly with multi-GPU)")
                _opt_state = torch.load(_optim_path, map_location='cpu', weights_only=False)
                optimizer.load_state_dict(_opt_state['optimizer'])
                if scheduler is not None and _opt_state.get('scheduler') is not None:
                    scheduler.load_state_dict(_opt_state['scheduler'])
    for epoch_id in range(_resume_epoch, num_epochs):
        model_logger._current_epoch = epoch_id
        # [NEW] resume 첫 epoch에서만 batch 위치까지 skip
        if epoch_id == _resume_epoch and _resume_step_in_epoch > 0:
            _active_dl = accelerator.skip_first_batches(dataloader, _resume_step_in_epoch)
            _step_offset = _resume_step_in_epoch
            if accelerator.is_main_process:
                print(f"[resume] Skipping {_resume_step_in_epoch} batches in epoch {epoch_id}")
        else:
            _active_dl = dataloader
            _step_offset = 0
        for _local_step, data in enumerate(tqdm(_active_dl)):
            _step_in_epoch = _step_offset + _local_step
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                # [NEW] grad_norm 로깅 + clipping
                _grad_norm = None
                if accelerator.sync_gradients:
                    _params = [p for p in model.parameters() if p.grad is not None]
                    if _params:
                        _max_gn = getattr(args, 'max_grad_norm', 0.0) if args is not None else 0.0
                        _clip_val = _max_gn if _max_gn > 0 else float('inf')
                        _grad_norm = torch.nn.utils.clip_grad_norm_(_params, max_norm=_clip_val).item()
                optimizer.step()
                # [NEW] dataloader 위치를 logger에 기록 (저장 시점에 custom_state.json으로 들어감)
                model_logger._current_step_in_epoch = _step_in_epoch + 1
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, grad_norm=_grad_norm)
                scheduler.step()
        if save_steps is None:
            # [NEW] epoch 끝 직후 저장: 다음 epoch 시작점이므로 step_in_epoch=0
            model_logger._current_step_in_epoch = 0
            model_logger._current_epoch = epoch_id + 1
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        # [NEW] 첫 resume epoch 이후로는 정상 진행
        _resume_step_in_epoch = 0
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
