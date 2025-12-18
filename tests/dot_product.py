# The dot product measures how much two vectors point
# in the same direction
# 1. Multiply each pair of corresponding elements
# 2. Sum those products

from typing import List


def dot(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("vectors must be of equal dimensions")

    sum = 0
    for x, y in zip(vec1, vec2):
        prd = x * y
        sum += prd

    return sum


if __name__ == "__main__":
    vec1 = [0.1, 0.2, 0.3]
    vec2 = [0.3, 0.2, 0.1]

    print(f"Dot product: {dot(vec1, vec2)}")


# Dot product has a problem, it is affected by vector magnitude
# In semantic search, we don't really care about magnitude
# Vectors that point in the same direction are similar
