import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from triton_barrier import blockwise_barrier


@triton.jit
def add_v4_bf16(a, b):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .v4 .b32 %acc, %tmp;
            mov.v4.b32  %acc, 0;
            mov.b64     {%acc.x, %acc.y}, $1;
            mov.b64     {%tmp.x, %tmp.y}, $2;
            add.bf16x2  %acc.x, %acc.x, %tmp.x;
            add.bf16x2  %acc.y, %acc.y, %tmp.y;
            mov.b64     $0, {%acc.x, %acc.y};
        }
        """,
        "=l,l,l",
        args=[a, b],
        dtype=(tl.uint64),
        is_pure=True,
        pack=1,
    )


@triton.jit
def one_shot_all_reduce_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size)
    pid = tl.program_id(axis=0)

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint64))
    block_start = pid * BLOCK_SIZE

    while block_start < (numel // NUMEL_PER_THREAD):
        offsets = (block_start + tl.arange(0, BLOCK_SIZE))
        mask = block_start + tl.arange(0, BLOCK_SIZE) < (numel // NUMEL_PER_THREAD)

        acc = tl.zeros((BLOCK_SIZE,), tl.uint64)
        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.uint64))
            val = tl.load(buffer_ptr + offsets, mask=mask)
            acc = add_v4_bf16(acc, val)

        tl.store(output_ptr + offsets, acc, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE


def one_shot_all_reduce(tensor: torch.Tensor):
    MAX_NUM_BLOCKS = 24
    NUM_WARPS = 16
    BLOCK_SIZE = NUM_WARPS * 32
    NUMEL_PER_THREAD = 4

    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert (
        tensor.numel() % NUMEL_PER_THREAD == 0
    ), "The number of elements must be 128-bit aligned."
    num_blocks = min(
        triton.cdiv(triton.cdiv(tensor.numel(), NUMEL_PER_THREAD), BLOCK_SIZE),
        MAX_NUM_BLOCKS,
    )

    symm_mem = _SymmetricMemory.rendezvous(tensor)
    output = torch.empty_like(tensor)

    one_shot_all_reduce_kernel[(num_blocks, 1, 1)](
        symm_mem.buffer_ptrs_dev,
        symm_mem.signal_pad_ptrs_dev,
        output,
        numel=tensor.numel(),
        rank=symm_mem.rank,
        world_size=symm_mem.world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        NUMEL_PER_THREAD=NUMEL_PER_THREAD,
        num_warps=NUM_WARPS,
    )
    return output


if __name__ == "__main__":
    """
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 -m symm_mem_recipes.triton_one_shot_all_reduce
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group_name = dist.group.WORLD.group_name
    enable_symm_mem_for_group(group_name)

    tensor = _SymmetricMemory.empty_strided_p2p(
        size=(8192,),
        stride=(1,),
        dtype=torch.bfloat16,
        device=device,
        group_name=group_name,
    ).fill_(rank)

    print("DEBUG", rank, world_size, tensor)
    output = one_shot_all_reduce(tensor)
    print("OUTPUT", rank, world_size, output)
    assert output.eq(world_size*(world_size - 1) // 2).all().item()

    dist.destroy_process_group()
