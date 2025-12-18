from typing import List


# Vector addition is useful for combining/mixing concepts
# I want a result that's like this and like that
def add_vectors(vec1: List[float], vec2: List[float]) -> List[float]:
    result_vec = []
    for x, y in zip(vec1, vec2):
        result_vec.append(x + y)

    return result_vec


# Vector substraction is useful for removing concepts
# I want a results that's like this but not that
def subtract_vectors(vec1: List[float], vec2: List[float]) -> List[float]:
    result_vec = []
    for x, y in zip(vec1, vec2):
        result_vec.append(x - y)

    return result_vec


if __name__ == "__main__":
    vec1 = [0.1, 0.2, 0.3]
    vec2 = [0.4, 0.5, 0.6]

    print(f"Sum: {add_vectors(vec1, vec2)}")
    print(f"Diff: {subtract_vectors(vec1, vec2)}")
