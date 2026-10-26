"""Keep reusable PyTorch allocator cache for the allocated single GPU process."""
import os
import torch


class GPUMemoryReservation:
    def __init__(self, minimum_mib=None):
        self.minimum_mib = int(os.environ.get('DATA_SELECT_MIN_GPU_MEMORY_MIB', '25600')) if minimum_mib is None else int(minimum_mib)
        if self.minimum_mib < 0:
            raise ValueError('DATA_SELECT_MIN_GPU_MEMORY_MIB must be nonnegative')
        self.minimum_bytes = self.minimum_mib * 1024 * 1024
        self.refills = 0
        self.observed = {}

    def ensure(self):
        reserved = torch.cuda.memory_reserved(0)
        if reserved < self.minimum_bytes:
            self.observed = dict(allocated_bytes=torch.cuda.memory_allocated(0), reserved_bytes=reserved, refills=self.refills)
            # Deletion returns storage to the reusable caching allocator. It does
            # not launch fill kernels or hold a competing permanent tensor.
            block = torch.empty(self.minimum_bytes - reserved, dtype=torch.uint8, device='cuda:0')
            del block
            self.refills += 1
        self.observed = dict(allocated_bytes=torch.cuda.memory_allocated(0),
            reserved_bytes=torch.cuda.memory_reserved(0),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0), refills=self.refills)
        if self.observed['reserved_bytes'] < self.minimum_bytes:
            raise RuntimeError(f'GPU allocator reservation below required floor: {self.observed}')
        return self.observed

    def metadata(self):
        return dict(minimum_mib=self.minimum_mib,
            policy='allocate deficit as empty uint8 then delete; keep reusable PyTorch allocator cache; never override empty_cache',
            observed=dict(self.observed))
