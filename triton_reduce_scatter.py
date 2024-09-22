import math
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from triton_barrier import blockwise_barrier


@triton.jit
def load_128(addrs, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32             %p0, $3, 1;
            @%p0 ld.global.v2.u64   {$0, $1}, [$2];
        }
        """,
        "=l,=l,l,r",
        args=[addrs, mask.to(tl.int32)],
        dtype=(tl.uint64, tl.uint64),
        is_pure=True,
        pack=1,
    )


@triton.jit
def add_v8_bf16(a_hi, a_lo, b_hi, b_lo):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .v4 .b32 %acc, %tmp;
            mov.v4.b32  %acc, 0;
            mov.b64     {%acc.x, %acc.y}, $2;
            mov.b64     {%acc.z, %acc.w}, $3;
            mov.b64     {%tmp.x, %tmp.y}, $4;
            mov.b64     {%tmp.z, %tmp.w}, $5;
            add.bf16x2  %acc.x, %acc.x, %tmp.x;
            add.bf16x2  %acc.y, %acc.y, %tmp.y;
            add.bf16x2  %acc.z, %acc.z, %tmp.z;
            add.bf16x2  %acc.w, %acc.w, %tmp.w;
            mov.b64     $0, {%acc.x, %acc.y};
            mov.b64     $1, {%acc.z, %acc.w};
        }
        """,
        "=l,=l,l,l,l,l",
        args=[a_hi, a_lo, b_hi, b_lo],
        dtype=(tl.uint64, tl.uint64),
        is_pure=True,
        pack=1,
    )


@triton.jit
def one_shot_reduce_scatter_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    output_ptr,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
):
    # blockwise_barrier(signal_pad_ptrs, None, rank, world_size)
    pid = tl.program_id(axis=0)

    per_rank_numel = numel // world_size

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint64))
    block_start = pid * BLOCK_SIZE

    while block_start < (per_rank_numel // NUMEL_PER_THREAD):
        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE) < per_rank_numel // NUMEL_PER_THREAD

        acc_hi = tl.zeros((BLOCK_SIZE,), tl.uint64)
        acc_lo = tl.zeros((BLOCK_SIZE,), tl.uint64)
        for i in range(world_size):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.uint64)) + rank * per_rank_numel // NUMEL_PER_THREAD * 2
            (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
            (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)

        tl.store(output_ptr + offsets + 0, acc_hi, mask=mask)
        tl.store(output_ptr + offsets + 1, acc_lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE


def one_shot_reduce_scatter(tensor: torch.Tensor):
    MAX_NUM_BLOCKS = 24
    NUM_WARPS = 16
    BLOCK_SIZE = NUM_WARPS * 32
    NUMEL_PER_THREAD = 8

    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert (
        tensor.numel() % (NUMEL_PER_THREAD * world_size) == 0
    ), "The number of elements must be 128-bit aligned."
    num_blocks = min(
        triton.cdiv(tensor.numel(), NUMEL_PER_THREAD * BLOCK_SIZE * world_size),
        MAX_NUM_BLOCKS,
    )

    per_rank_size = int(tensor.numel() / world_size)

    symm_mem = _SymmetricMemory.rendezvous(tensor)
    local_tensor_slice = tensor[per_rank_size*rank:per_rank_size*(rank+1)]
    output = torch.empty_like(local_tensor_slice)

    one_shot_reduce_scatter_kernel[(num_blocks, 1, 1)](
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
    # """
    # torchrun \
    # --nnodes 1 --nproc-per-node 8 \
    # --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    # --no_python python3 -m symm_mem_recipes.triton_reduce_scatter
    # """
    # rank = int(os.environ["RANK"])
    # local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])

    # device = torch.device(f"cuda:{local_rank}")
    # torch.cuda.set_device(device)
    # dist.init_process_group("nccl")
    # group_name = dist.group.WORLD.group_name
    # enable_symm_mem_for_group(group_name)

    # torch.manual_seed(rank)

    # tensor = _SymmetricMemory.empty_strided_p2p(
    #     size=(8192,),
    #     stride=(1,),
    #     dtype=torch.bfloat16,
    #     device=device,
    #     group_name=group_name,
    # ).copy_(torch.randn(8192, dtype=torch.bfloat16))

    # answer = torch.zeros(8192, dtype=torch.bfloat16)
    # for i in range(world_size):
    #     torch.manual_seed(i)
    #     answer += torch.randn(8192, dtype=torch.bfloat16)

    # # print("DEBUG", rank, world_size, tensor.reshape(world_size, -1))
    # if rank == 0:
    #     print("REFERENCE", answer)
    # output = one_shot_reduce_scatter(tensor)
    # print(f"OUTPUT {rank} {world_size} {output}")

    # NUMEL_PER_THREAD = 8
    # per_rank_size = math.ceil(tensor.numel() / NUMEL_PER_THREAD / world_size) * NUMEL_PER_THREAD

    # assert (output.cpu() == answer[per_rank_size * rank : per_rank_size * (rank + 1)]).all()

    # dist.destroy_process_group()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group_name = dist.group.WORLD.group_name
    enable_symm_mem_for_group(group_name)

    torch.manual_seed(rank)

    size = 256*1024*1024

    tensor = _SymmetricMemory.empty_strided_p2p(
        size=(size,),
        stride=(1,),
        dtype=torch.bfloat16,
        device=device,
        group_name=group_name,
    ).copy_(torch.randn(size, dtype=torch.bfloat16))

    answer = torch.zeros(size, dtype=torch.bfloat16)
    for i in range(world_size):
        torch.manual_seed(i)
        answer += torch.randn(size, dtype=torch.bfloat16)

    if rank == 0:
        print("REFERENCE", answer)

    output = one_shot_reduce_scatter(tensor)
    print(f"OUTPUT {rank} {world_size} {output}")

    NUMEL_PER_THREAD = 8
    per_rank_size = math.ceil(tensor.numel() / NUMEL_PER_THREAD / world_size) * NUMEL_PER_THREAD

    assert torch.allclose(output.cpu(), answer[per_rank_size * rank : per_rank_size * (rank + 1)])

    while size >= 2048:
        torch.cuda.synchronize()

        REPS = 10

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPS):
            output = one_shot_reduce_scatter(tensor[:size])
        end.record()
        torch.cuda.synchronize()
        if rank == 0:
            print(f"triton {size * 2} {size * 2 / start.elapsed_time(end) / 1e6 * REPS} MB/s")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        indices = torch.arange(per_rank_size * rank, per_rank_size * (rank + 1)).cuda()
        dist.reduce_scatter_tensor(output, tensor[:size])  # Warmup
        start.record()
        for _ in range(REPS):
            dist.reduce_scatter_tensor(output, tensor[:size])
        end.record()
        torch.cuda.synchronize()
        if rank == 0:
            print(f"nccl {size * 2} {size * 2 / start.elapsed_time(end) / 1e6 * REPS} MB/s")

        size //= 2

    dist.destroy_process_group()
