#! /usr/bin/env python3

# Writes the small datasets used by this test, in the SOSD format
# (a uint64 count followed by the sorted keys).

import math
import struct

N = 200000


def write(fn, keys, fmt):
    keys = sorted(set(keys))
    with open(fn, "wb") as f:
        f.write(struct.pack("<Q", len(keys)))
        f.write(struct.pack("<%d%s" % (len(keys), fmt), *keys))


# Consecutive integers: the leaf models fit exactly (error 0).
write("data_dense_uint64", [1000 + i for i in range(N)], "Q")
write("data_dense_uint32", [1000 + i for i in range(N)], "I")

# Evenly spaced keys far from 0: the cubic top model is a straight line.
write("data_line_uint64", [10**12 + 1000 * i for i in range(N)], "Q")

# Keys whose CDF is the cubic 3t^2 - 2t^3 on [0.15, 0.85] * 2^64: the
# cubic top model has its extrema at the smallest and the largest key and
# turns back outside of them, to leaf models in the middle. Two gaps leave
# some of those leaf models empty.
lo, hi = 0.15 * 2**64, 0.85 * 2**64
write("data_turn_uint64",
      [int(lo + (hi - lo) * (0.5 - math.sin(math.asin(1 - 2 * i / (N - 1)) / 3)))
       for i in range(N) if not (0.05 * N <= i < 0.10 * N or 0.90 * N <= i < 0.95 * N)],
      "Q")

# Evenly spaced keys in [2^62, 3 * 2^62): the keys have no common prefix,
# and the radix table entries after the largest key are never set.
write("data_radix_uint64", [2**62 + (2**63 // N) * i for i in range(N)], "Q")

# Two clusters of 32-bit keys, at 2^20 and at 2^31: the radix top model
# (bits 31..22 for 1024 leaf models) leaves most leaf models empty, and
# 64-bit keys of 2^32 and above reach them too.
write("data_split_uint32",
      [2**20 + 16 * i for i in range(N // 2)] + [2**31 + 16 * i for i in range(N // 2)], "I")

# Two dense clusters of keys, from 2^40 and from 1.75 * 2^40 (each about
# 1.6 million wide), with ten widely spaced keys between them, which share
# one leaf model. Radix tables ignore the common prefix and index the low
# 40 bits, so keys outside of the data range (below 2^40 or from 2^41 on)
# can reach that leaf model too.
write("data_prefix_uint64",
      [2**40 + 16 * i for i in range(N // 2)]
      + [2**40 + 2**38 + 2**35 * j for j in range(10)]
      + [2**40 + 3 * 2**38 + 16 * i for i in range(N // 2 - 10)], "Q")
