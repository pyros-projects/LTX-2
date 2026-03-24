from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import torch
from torch import nn


def _clean_memory_on_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _move_linear_weights_to_device(layer: nn.Module, device: torch.device) -> None:
    for module in layer.modules():
        if (
            hasattr(module, "weight")
            and module.weight is not None
            and module.__class__.__name__.endswith("Linear")
        ):
            module.weight.data = module.weight.data.to(device, non_blocking=device.type != "cpu")


def _swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module) -> None:
    del device
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules(), strict=False):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append(
                (
                    module_to_cpu,
                    module_to_cuda,
                    module_to_cpu.weight.data,
                    module_to_cuda.weight.data,
                )
            )

    for module_to_cpu, _module_to_cuda, cuda_data_view, _cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=False)

    for module_to_cpu, module_to_cuda, cuda_data_view, _cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=False)
        module_to_cuda.weight.data = cuda_data_view
        module_to_cpu.weight.data = module_to_cpu.weight.data


class Offloader:
    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
    ) -> None:
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.use_pinned_memory = use_pinned_memory
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures: dict[int, object] = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None
        self.staging_buffer_a: list[torch.Tensor] | None = None
        self.staging_buffer_b: list[torch.Tensor] | None = None
        self.pinned_buffer: list[torch.Tensor] | None = None

    def swap_weight_devices_cuda(  # noqa: PLR0912
        self,
        device: torch.device,
        layer_to_cpu: nn.Module,
        layer_to_cuda: nn.Module,
    ) -> torch.cuda.Event | None:
        assert self.stream is not None
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        modules_to_cpu = dict(layer_to_cpu.named_modules())
        weight_swap_jobs = []

        for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
            if (
                hasattr(module_to_cuda, "weight")
                and module_to_cuda.weight is not None
                and module_to_cuda.__class__.__name__.endswith("Linear")
            ):
                module_to_cpu = modules_to_cpu.get(module_to_cuda_name)
                if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                    weight_swap_jobs.append(
                        (module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data)
                    )
                elif module_to_cuda.weight.data.device.type != device.type:
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

        torch.cuda.current_stream(device=device).synchronize()

        if not self.use_pinned_memory:
            with torch.cuda.stream(self.stream):
                if self.staging_buffer_a is None or self.staging_buffer_b is None:
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]

                event_b = None
                for staging_a, staging_b, (_module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                    self.staging_buffer_a,
                    self.staging_buffer_b,
                    weight_swap_jobs,
                    strict=False,
                ):
                    event_a = torch.cuda.Event()
                    staging_a.copy_(cuda_data_view.data, non_blocking=True)
                    event_a.record(self.stream)

                    if event_b is not None:
                        event_b.synchronize()

                    staging_b.copy_(module_to_cuda.weight.data)
                    event_a.synchronize()

                    event_b = torch.cuda.Event()
                    cuda_data_view.copy_(staging_b, non_blocking=True)
                    event_b.record(self.stream)
                    cpu_data_view.copy_(staging_a)

            for (_module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in weight_swap_jobs:
                module_to_cuda.weight.data = cuda_data_view
                _module_to_cpu.weight.data = cpu_data_view

            return event_b

        if self.pinned_buffer is None:
            with torch.cuda.stream(self.stream):
                self.pinned_buffer = [
                    torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                    for _, _, cuda_data_view, _ in weight_swap_jobs
                ]
            self.stream.synchronize()

        released_pinned_buffer = []
        events = [torch.cuda.Event() for _ in weight_swap_jobs]

        for event, module_pin_buf, (_module_to_cpu, _module_to_cuda, cuda_data_view, _cpu_data_view) in zip(
            events,
            self.pinned_buffer,
            weight_swap_jobs,
            strict=False,
        ):
            with torch.cuda.stream(self.stream):
                module_pin_buf.copy_(cuda_data_view, non_blocking=True)
                event.record(self.stream)

        for event, (_module_to_cpu, _module_to_cuda, cuda_data_view, cpu_data_view) in zip(
            events,
            weight_swap_jobs,
            strict=False,
        ):
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(event)
                cuda_data_view.copy_(cpu_data_view, non_blocking=True)

        for module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
            self.pinned_buffer,
            weight_swap_jobs,
            strict=False,
        ):
            module_to_cuda.weight.data = cuda_data_view
            module_to_cpu.weight.data = module_pin_buf
            released_pinned_buffer.append(cpu_data_view)

        if released_pinned_buffer and not released_pinned_buffer[0].is_pinned():
            with torch.cuda.stream(self.stream):
                released_pinned_buffer = [
                    torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                    for _, _, cuda_data_view, _ in weight_swap_jobs
                ]
        self.pinned_buffer = released_pinned_buffer
        return self.stream.record_event()

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module) -> torch.cuda.Event | None:
        if self.cuda_available:
            return self.swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)

        _swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)
        return None

    def _submit_move_blocks(
        self,
        blocks: list[nn.Module] | nn.ModuleList,
        block_idx_to_cpu: int,
        block_idx_to_cuda: int,
    ) -> None:
        def move_blocks(
            bidx_to_cpu: int,
            block_to_cpu: nn.Module,
            bidx_to_cuda: int,
            block_to_cuda: nn.Module,
        ) -> tuple[int, int, torch.cuda.Event | None]:
            if self.cuda_available:
                dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
                torch.cuda.set_device(dev)
            sync_event = self.swap_weight_devices(block_to_cpu, block_to_cuda)
            return bidx_to_cpu, bidx_to_cuda, sync_event

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks,
            block_idx_to_cpu,
            blocks[block_idx_to_cpu],
            block_idx_to_cuda,
            blocks[block_idx_to_cuda],
        )

    def _wait_blocks_move(self, block_idx: int) -> None:
        if block_idx not in self.futures:
            return

        future = self.futures.pop(block_idx)
        _block_idx_to_cpu, bidx_to_cuda, sync_event = future.result()
        if block_idx != bidx_to_cuda:
            raise ValueError(f"Block index mismatch: {block_idx} != {bidx_to_cuda}")

        if self.cuda_available and sync_event is not None:
            torch.cuda.current_stream(device=self.device).wait_event(sync_event)


class ModelOffloader(Offloader):
    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module] | nn.ModuleList,
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        use_pinned_memory: bool = False,
    ) -> None:
        super().__init__(block_type, num_blocks, blocks_to_swap, device, use_pinned_memory)
        self.supports_backward = supports_backward
        self.forward_only = not supports_backward
        self.remove_handles: list[torch.utils.hooks.RemovableHandle] = []

        if self.supports_backward:
            for block_index, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, block_index)
                if hook is not None:
                    self.remove_handles.append(block.register_full_backward_hook(hook))

    def set_forward_only(self, forward_only: bool) -> None:
        for block_idx in list(self.futures):
            self._wait_blocks_move(block_idx)
        self.forward_only = forward_only

    def __del__(self) -> None:
        for handle in self.remove_handles:
            handle.remove()

    def create_backward_hook(
        self,
        blocks: list[nn.Module] | nn.ModuleList,
        block_index: int,
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], None] | None:
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = 0 < num_blocks_propagated <= self.blocks_to_swap
        waiting = 0 < block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            del module, grad_input, grad_output
            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module] | nn.ModuleList) -> None:
        if self.blocks_to_swap == 0:
            return

        for block in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            block.to(self.device)
            _move_linear_weights_to_device(block, self.device)

        cpu_device = torch.device("cpu")
        for block in blocks[self.num_blocks - self.blocks_to_swap :]:
            block.to(self.device)
            _move_linear_weights_to_device(block, cpu_device)

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int) -> None:
        if self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module] | nn.ModuleList, block_idx: int) -> None:
        if self.blocks_to_swap == 0:
            return

        if not self.forward_only:
            if block_idx >= self.blocks_to_swap:
                return
            block_idx_to_cpu = block_idx
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda %= self.num_blocks
            self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            return

        block_idx_to_cpu = block_idx
        if self.blocks_to_swap < (self.num_blocks // 2):
            if self.blocks_to_swap <= block_idx < self.num_blocks - self.blocks_to_swap:
                return
            if block_idx < self.blocks_to_swap:
                block_idx_to_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            else:
                block_idx_to_cuda = block_idx - (self.num_blocks - self.blocks_to_swap)
        else:
            block_idx_to_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks

        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
