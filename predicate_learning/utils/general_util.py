import numpy as np
import torch

import time
from termcolor import colored
from dataclasses import dataclass
from typing import Optional


############# Environments ############
def l2_distance_np(x, y):
    return np.linalg.norm(x - y)


def l2_distance_torch(x, y):
    return torch.norm(x - y)


def check_tensor_diff(x, y, tol=1e-2):
    return l2_distance_torch(x, y) > tol


def check_edge(pos_1, size_1, pos_2, size_2, threshood=6):
    x_min_1 = pos_1[0] - size_1[0] / 2
    x_max_1 = pos_1[0] + size_1[0] / 2
    y_min_1 = pos_1[1]
    y_max_1 = pos_1[1] + size_1[1]

    x_min_2 = pos_2[0] - size_2[0] / 2
    x_max_2 = pos_2[0] + size_2[0] / 2
    y_min_2 = pos_2[1]
    y_max_2 = pos_2[1] + size_2[1]

    if (
        x_min_1 - x_max_2 <= threshood
        and x_min_2 - x_max_1 <= threshood
        and y_min_1 - y_max_2 <= threshood
        and y_min_2 - y_max_1 <= threshood
    ):
        return True
    else:
        return False


def check_edge_gripper(pos, size, pos_gripper, threshood=6):
    pos = pos.clone()
    pos[1] += size[1] / 2

    if l2_distance_torch(pos, pos_gripper) <= threshood:
        return True
    else:
        return False


############# Timer ###############
class Timer(object):
    def __init__(self, logger, color="black"):
        self.logger = logger
        self.last_time = []
        self.event = []
        self.color = color

    def tic(self, event: str):
        self.last_time.append(time.time())
        self.event.append(event)

    def toc(self):
        elapsed = None
        if len(self.last_time) > 0:
            current_time = time.time()
            elapsed = current_time - self.last_time.pop()

            self.logger.debug(
                colored(
                    f"Elapsed time: {elapsed} seconds for {self.event.pop()}",
                    self.color,
                )
            )
        return elapsed


############# Bit Manipulation ###############
@dataclass
class BitMaskPair:
    pos_mask: Optional[int] = 0
    val_mask: Optional[int] = 0
    num_bits: Optional[int] = 0

    def value_inverted(self):
        return self.__class__(self.pos_mask, ~self.val_mask)

    def with_pos_mask(self, pos_mask):
        return self.__class__(pos_mask, self.val_mask)

    def set(self, set_pair):
        # Set position
        self.pos_mask |= set_pair.pos_mask

        # Set value
        # set specific positions to 0, while keep other positions unchanged
        self.val_mask &= ~set_pair.pos_mask
        # set specific positions to 1 if val_mask is 1
        self.val_mask |= set_pair.pos_mask & set_pair.val_mask

    def set_free(self, set_pair):
        # Get free positions to set
        pos_mask_to_set = set_pair.pos_mask & ~self.pos_mask

        # Set
        self.set(set_pair.with_pos_mask(pos_mask_to_set))
    
    def check_no_conflict(self, check_pair):
        # Get overlapped positions
        pos_mask_overlap = check_pair.pos_mask & self.pos_mask

        # For self, get values at specific positions while set other bits as 0
        get_val = pos_mask_overlap & self.val_mask

        # For set pair, get values at specific positions while set other bits as 0
        check_val = pos_mask_overlap & check_pair.val_mask

        # Check if get_val == check_val
        return get_val ^ check_val == 0

    def to_tensors(self, device):
        assert self.num_bits > 0
        mask = int2bin(self.pos_mask, self.num_bits, device).bool()
        labels = int2bin(self.val_mask, self.num_bits, device).long()
        return labels, mask

    @classmethod
    def from_tensors(cls, labels, mask=None):
        if mask is None:
            pos_mask = 0
        else:
            pos_mask = int(bin2int(mask).item())

        val_mask = int(bin2int(labels).item())
        num_bits = labels.shape[-1]
        return cls(pos_mask, val_mask, num_bits)


def int2bin(x, num_bits, device=None):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    device = device or x.device
    if isinstance(x, int):
        x = torch.tensor(x, device=device)
    mask = 2 ** torch.arange(num_bits - 1, -1, -1).to(device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2int(b):
    num_bits = b.shape[-1]
    mask = 2 ** torch.arange(num_bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


# def set_bit_masks(base_pair, set_pair):
#     # Set position
#     base_pair.pos_mask |= set_pair.pos_mask

#     # Set value
#     # set specific positions to 0, while keep other positions unchanged
#     base_pair.value_mask &= ~set_pair.pos_mask
#     # set specific positions to 1 if val_mask is 1
#     base_pair.value_mask |= set_pair.pos_mask & set_pair.val_mask
