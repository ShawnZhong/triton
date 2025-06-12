import os
import torch
import torch.distributed as dist
import triton_kernels.routing
from triton_kernels.routing import RoutingData, GatherIndx, ScatterIndx, compute_expt_data
from triton_kernels.topk import topk
from typing import Tuple


def _is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup() -> Tuple[int, int]:
    if _is_distributed_launch():
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", world_size=world_size, device_id=torch.device(local_rank))
    else:
        world_size = 1
        local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, world_size


def cleanup():
    if _is_distributed_launch():
        dist.barrier()
        dist.destroy_process_group()
    else:
        pass


def broadcast(x: torch.Tensor, src: int = 0, groups: list = None, group_idx: int = None) -> torch.Tensor:
    if _is_distributed_launch():
        if x.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            x = x.to(torch.float16)
        group = None
        if groups:
            groups = [dist.new_group(group) for group in groups]
            dist.barrier()
            group = groups[group_idx]
        dist.broadcast(x, src=src, group=group)
        return x
    else:
        return x


def all_gather(x: torch.Tensor, dim=0) -> torch.Tensor:
    if _is_distributed_launch():
        world_size = dist.get_world_size()
        x_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(x_list, x)
        return torch.cat(x_list, dim=dim)
    else:
        return x


def reduce_scatter(x: torch.Tensor, gpu_idx: torch.Tensor = None, dim=0) -> torch.Tensor:
    if _is_distributed_launch():
        world_size = dist.get_world_size()
        if gpu_idx is not None:
            assert dim == 0, "gpu_idx only works with dim=0"
            # Use all to all to simulate reduce scatter
            input_split_sizes = []
            for i in range(world_size):
                input_split_sizes.append((gpu_idx == i).sum().item())
            sizes = all_gather(torch.tensor(input_split_sizes, device=x.device), dim=0)
            output_split_sizes = sizes[:, dist.get_rank()].tolist()
            all_to_all_single(x, input_split_sizes=output_split_sizes, output_split_sizes=input_split_sizes)
        else:
            x_list = list(x.chunk(world_size, dim=dim))
            # build output shape
            shape = x_list[0].shape
            # reduce scatter into the single tensor
            # check if dtype is fp8, then convert it to float16 before reducing
            if x.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
                x_list = [x.to(torch.float16) for x in x_list]
                out = x.new_empty(shape, dtype=torch.float16)
            else:
                out = x.new_empty(shape, dtype=x.dtype)
            dist.reduce_scatter(out, x_list)
            return out
    else:
        return x


def all_to_all_single(x: torch.Tensor, output_split_sizes: list[int], input_split_sizes: list[int]) -> torch.Tensor:
    if _is_distributed_launch():
        output_dim0 = sum(output_split_sizes)
        output = torch.empty((output_dim0, *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_to_all_single(output, x, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes)
        return output
    else:
        return x


def routing(logits, n_expts_act, sm_first=False, expt_indx=None, n_rows=None, EP=1,
            TP=1) -> Tuple[RoutingData, GatherIndx, ScatterIndx, torch.Tensor]:
    if _is_distributed_launch():
        assert expt_indx is None
        _, n_expts_tot = logits.shape
        # We need to use the same topk as triton_kernels because torch's topk
        # does not have the same tie-breaking behavior as triton_kernels.
        if sm_first:
            logits = torch.softmax(logits, dim=-1)
        expt_scal, expt_indx, _ = topk(logits, n_expts_act, apply_softmax=sm_first, n_rows=n_rows)
        expt_indx = expt_indx.int()
        # Sort each token's selections by expert
        expt_indx, sort_indices = torch.sort(expt_indx, dim=1, stable=True)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)
        chunk_size = n_expts_tot // EP
        ep_indx = dist.get_rank() // TP
        gpu_idx = expt_indx // chunk_size
        if EP > 1:
            # Distributed-EP
            # figure out how many tokens are assigned to each expert
            input_split_sizes = []
            for i in range(dist.get_world_size()):
                input_split_sizes.append((gpu_idx == i).sum().item())
            sizes = all_gather(torch.tensor(input_split_sizes, device=logits.device), dim=0)
            output_split_sizes = sizes[:, dist.get_rank()].tolist()
            expt_scal = all_to_all_single(expt_scal, output_split_sizes, input_split_sizes)
            expt_indx = all_to_all_single(expt_indx, output_split_sizes, input_split_sizes)
            expt_indx -= ep_indx * chunk_size
        else:
            # Distributed-DP
            expt_scal = all_gather(expt_scal, dim=0)
            expt_indx = all_gather(expt_indx, dim=0)
        # flatten topk data
        expt_scal = expt_scal.reshape(-1)
        expt_indx = expt_indx.reshape(-1).to(torch.int32)
        # sort by expert_id so experts are contiguous for the matmul
        # For example:
        # expt_indx: [expt0 => row4, row1, row0, ..., expt1 => row2, row3, ..., ...]
        # topk_indx: [2 (row0), 1 (row1), 3 (row2), 4 (row3), 5 (row4), ...]
        expt_indx, topk_indx = torch.sort(expt_indx, stable=True)
        gate_indx = torch.argsort(topk_indx, stable=True)
        gate_scal = expt_scal[topk_indx]
        # histogram of tokens over experts
        hist = torch.histc(expt_indx, bins=chunk_size, min=0, max=chunk_size)
        # pack the matmul data structure
        gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
        scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
        n_gates = topk_indx.numel()
        expt_data = compute_expt_data(hist, n_expts_tot // EP, n_gates)
        return RoutingData(gate_scal, hist, n_expts_tot // EP, n_expts_act,
                           expt_data=expt_data), gather_indx, scatter_indx, gpu_idx
    else:
        return *triton_kernels.routing.routing(logits, n_expts_act, sm_first, expt_indx, EP, n_rows), None
