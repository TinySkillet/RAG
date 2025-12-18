from typing import List

from tests.dot_product import dot


def euclidean_norm(vec: List[float]) -> float:
    total = 0.0
    for x in vec:
        total += x**2

    return total**0.5


# cosine similarity measures the cosine of angle between two vectors
# value ranges from -1.0 to 1.0
# 1.0 means vectors point exactly in the same direction (perfectly similar)
# 0.0 means vectors are perpendicular to one another (no similarity)
# -1.0 means vectors are in opposite directions (perfectly dissimilar)
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("vectors must be of equal dimensions")

    dot_prd = dot(vec1, vec2)
    mag_vec1 = euclidean_norm(vec1)
    mag_vec2 = euclidean_norm(vec2)

    cos_sim = dot_prd / (mag_vec1 * mag_vec2)
    return cos_sim


if __name__ == "__main__":
    vec1 = [0.1, 0.2, 0.3]
    vec2 = [0.3, 0.2, 0.1]

    print(f"Cosine similarity: {cosine_similarity(vec1, vec2)}")
