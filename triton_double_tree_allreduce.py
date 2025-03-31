import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from tree import get_parents_and_children
from triton_barrier import blockwise_barrier


@triton.jit
def get_flat_tid():
    return tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %tmp32_<2>;

            mov.u32     %tmp32_0, %tid.z;
            mov.u32     %tmp32_1, %ntid.y;
            mul.lo.u32  %tmp32_0, %tmp32_0, %tmp32_1; // tid.z * ntid.y
            mov.u32     %tmp32_1, %ntid.x;
            mul.lo.u32  $0, %tmp32_0, %tmp32_1;       // $0 = tid.z * ntid.y * ntid.x
            mov.u32     %tmp32_0, %tid.y;
            mov.u32     %tmp32_1, %ntid.x;
            mul.lo.u32  %tmp32_0, %tmp32_0, %tmp32_1; // tid.y * ntid.x
            add.u32     $0, $0, %tmp32_0;             // $0 += tid.y * ntid.x
            mov.u32     %tmp32_0, %tid.x;
            add.u32     $0, $0, %tmp32_0;             // $0 += tid.x
        }
        """,
        "=r",
        [],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def blockwise_barrier_double_tree(
    signal_pad_ptrs,
    block_id,
    send1_rank, send2_rank,
    wait1_rank, wait2_rank,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    # UP_RANKS = [
    #     # Parent ranks up the tree
    #     [-1, 2, 4, 2, 0, 6, 4, 6], # tree 0
    #     [1, 3, 1, 7, 5, 3, 5, -1], # tree 1
    # ]

    if block_id is None:
        block_id = (
            tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
            + tl.program_id(1) * tl.num_programs(0)
            + tl.program_id(0)
        )
    flat_tid = get_flat_tid()

    remote_ranks = tl.cat(tl.full((1,), (send1_rank), tl.int32), tl.full((1,), (send2_rank), tl.int32))
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * WORLD_SIZE + RANK

    remote_ranks = tl.cat(tl.full((1,), (wait1_rank), tl.int32), tl.full((1,), (wait2_rank), tl.int32))
    local_signal_pad_addr = tl.load(signal_pad_ptrs + RANK).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * WORLD_SIZE + remote_ranks

    if flat_tid < WORLD_SIZE:
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;

                wait_signal:
                    // No need to acquire here since all threads will
                    // acquire this location after the barrier.
                    atom.global.sys.cas.b32 %tmp32_0, [$2], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;

                barrier_end:
            }
            """,
            "=r, l, l",
            [send_addrs, wait_addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )

    tl.inline_asm_elementwise(
        "ld.acquire.sys.global.u32 $0, [$1];",
        "=r, l",
        [local_signal_pad_addr + send1_rank],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )

    tl.inline_asm_elementwise(
        "ld.acquire.sys.global.u32 $0, [$1];",
        "=r, l",
        [local_signal_pad_addr + send2_rank],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


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
def triton_wait(wait_addrs):
    flat_tid = get_flat_tid()
    if flat_tid == 0:
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    // No need to acquire here since all threads will
                    // acquire this location after the barrier.
                    atom.global.sys.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;

                barrier_end:
            }
            """,
            "=r, l",
            [wait_addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )

@triton.jit
def triton_send(send_addrs):
    flat_tid = get_flat_tid()
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )    
    if flat_tid == 0:
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;

                barrier_end:
            }
            """,
            "=r, l",
            [send_addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )

@triton.jit
def double_tree_all_reduce_kernel(
    buffer_ptrs,
    signal_pad_ptrs,
    output_ptr,
    tree0_parent, tree0_child0, tree0_child1, tree1_parent, tree1_child0, tree1_child1,
    numel: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size)
    block_id = (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )

    pid = tl.program_id(axis=0)

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint64))
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    block_start = pid * BLOCK_SIZE

    if pid < tl.num_programs(axis=0) // 2:
        tree_child0 = tree0_child0
        tree_child1 = tree0_child1
        tree_parent = tree0_parent
    else:
        tree_child0 = tree1_child0
        tree_child1 = tree1_child1
        tree_parent = tree1_parent

    if tree_child0 != -1 :
        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint32))
        wait_addrs = local_signal_pad_addr + block_id * world_size + tree_child0
        triton_wait(wait_addrs)
    if tree_child1 != -1 :
        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint32))
        wait_addrs = local_signal_pad_addr + block_id * world_size + tree_child1
        triton_wait(wait_addrs)

    while block_start < (numel // NUMEL_PER_THREAD):
        # Each thread processes 128 bits. Since Triton doesn't yet natively
        # support 128-bit dtypes, we achieve this by having each thread process
        # two 64-bit elements.

        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE) < numel // NUMEL_PER_THREAD

        acc_hi = tl.zeros((BLOCK_SIZE,), tl.uint64)
        acc_lo = tl.zeros((BLOCK_SIZE,), tl.uint64)
        if tree_child0 != -1:
            buffer_ptr = tl.load(buffer_ptrs + tree_child0).to(tl.pointer_type(tl.uint64))
            (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
            (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        # All the computation in else Region will be eliminated by DSE pass, but it's necessary in triton's frontend parser now
        else :
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.uint64))
            hi = tl.zeros((BLOCK_SIZE,), tl.uint64)
            lo = tl.zeros((BLOCK_SIZE,), tl.uint64)
            (acc_hi, acc_lo) = (acc_hi, acc_lo)

        if tree_child1 != -1:
            buffer_ptr = tl.load(buffer_ptrs + tree_child1).to(tl.pointer_type(tl.uint64))
            (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
            (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        else :
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.uint64))
            hi = tl.zeros((BLOCK_SIZE,), tl.uint64)
            lo = tl.zeros((BLOCK_SIZE,), tl.uint64)
            (acc_hi, acc_lo) = (acc_hi, acc_lo)
 

        buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.uint64))
        (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
        (acc_hi, acc_lo) = add_v8_bf16(acc_hi, acc_lo, hi, lo)
        
        tl.store(buffer_ptr + offsets + 0, acc_hi, mask=mask)
        tl.store(buffer_ptr + offsets + 1, acc_lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE
    
    if tree_parent != -1:
        remote_signal_pad_addrs = tl.load(signal_pad_ptrs + tree_parent).to(tl.pointer_type(tl.uint32))
        send_addrs = remote_signal_pad_addrs + block_id * world_size + rank
        triton_send(send_addrs)

        local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(tl.pointer_type(tl.uint32))
        wait_addrs = local_signal_pad_addr + block_id * world_size + tree_parent
        triton_wait(wait_addrs)

    block_start = pid * BLOCK_SIZE
    while block_start < (numel // NUMEL_PER_THREAD):
        # Each thread processes 128 bits. Since Triton doesn't yet natively
        # support 128-bit dtypes, we achieve this by having each thread process
        # two 64-bit elements.

        offsets = (block_start + tl.arange(0, BLOCK_SIZE)) * 2
        mask = block_start + tl.arange(0, BLOCK_SIZE) < numel // NUMEL_PER_THREAD

        if tree_parent != -1:
            buffer_ptr = tl.load(buffer_ptrs + tree_parent).to(tl.pointer_type(tl.uint64))
            (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.uint64))
            tl.store(buffer_ptr + offsets + 0, hi, mask=mask)
            tl.store(buffer_ptr + offsets + 1, lo, mask=mask)
            tl.store(output_ptr + offsets + 0, hi, mask=mask)
            tl.store(output_ptr + offsets + 1, lo, mask=mask)
        else:
            buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.uint64))
            (hi, lo) = load_128(buffer_ptr + offsets, mask=mask)
            tl.store(output_ptr + offsets + 0, hi, mask=mask)
            tl.store(output_ptr + offsets + 1, lo, mask=mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    if tree_child0 != -1:
        remote_signal_pad_addrs = tl.load(signal_pad_ptrs + tree_child0).to(tl.pointer_type(tl.uint32))
        send_addrs = remote_signal_pad_addrs + block_id * world_size + rank
        triton_send(send_addrs)
    if tree_child1 != -1:
        remote_signal_pad_addrs = tl.load(signal_pad_ptrs + tree_child1).to(tl.pointer_type(tl.uint32))
        send_addrs = remote_signal_pad_addrs + block_id * world_size + rank
        triton_send(send_addrs)

def double_tree_all_reduce(tensor: torch.Tensor):
    MAX_NUM_BLOCKS = 24
    NUM_WARPS = 4
    BLOCK_SIZE = NUM_WARPS * 32
    NUMEL_PER_THREAD = 8

    assert tensor.dtype == torch.bfloat16, "Only bfloat16 is supported for now."
    assert (
        tensor.numel() % NUMEL_PER_THREAD == 0
    ), "The number of elements must be 128-bit aligned."
    num_blocks = min(
        triton.cdiv(triton.cdiv(tensor.numel(), NUMEL_PER_THREAD), BLOCK_SIZE),
        MAX_NUM_BLOCKS,
    )
    assert num_blocks % 2 == 0, "Better strike a balance between two trees"
    symm_mem = _SymmetricMemory.rendezvous(tensor)
    output = torch.empty_like(tensor)

    tree0_parent, tree0_child0, tree0_child1, tree1_parent, tree1_child0, tree1_child1 = get_parents_and_children(8, rank)
    double_tree_all_reduce_kernel[(num_blocks, 1, 1)](
        symm_mem.buffer_ptrs_dev,
        symm_mem.signal_pad_ptrs_dev,
        output,
        tree0_parent, tree0_child0, tree0_child1, tree1_parent, tree1_child0, tree1_child1,
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
    touch triton_double_tree_allreduce.py && python3 -m torch.distributed.launch --nnodes 1 --nproc-per-node 8 triton_double_tree_allreduce.py
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group_name = dist.group.WORLD.group_name
    enable_symm_mem_for_group(group_name)

    torch.manual_seed(rank)

    size = 2048*1024

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
    output = double_tree_all_reduce(tensor)
    print(f"OUTPUT {rank} {world_size} {output}")

    if rank == 0:
        print("REFERENCE", answer)
    assert torch.allclose(output.cpu(), answer, rtol=1e-1, atol=1e-01)

    torch.cuda.synchronize()

    REPS = 10

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.all_reduce(tensor[:size])  # Warmup
    start.record()
    for _ in range(REPS):
        dist.all_reduce(tensor)
    end.record()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"nccl {size * 2} {size * 2 / start.elapsed_time(end) / 1e6 * REPS} MB/s")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    double_tree_all_reduce(tensor[:size])
    start.record()
    for _ in range(REPS):
        double_tree_all_reduce(tensor[:size])
    end.record()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"triton {size * 2} {size * 2 / start.elapsed_time(end) / 1e6 * REPS} MB/s")

    dist.destroy_process_group()
