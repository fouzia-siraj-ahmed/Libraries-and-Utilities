import numpy as np
from scipy.spatial.distance import cityblock, cosine, minkowski, hamming

# Define points
p = np.array([3, 4])
q = np.array([0, 0])

# Euclidean distance
euclidean_distance = np.linalg.norm(p - q)

# Manhattan distance
manhattan_distance = cityblock(p, q)

# Cosine similarity
cosine_similarity = 1 - cosine(p, q)  # Subtract from 1 to get similarity

# Minkowski distance (p=3)
minkowski_distance = minkowski(p, q, 3)

# Hamming distance (binary example)
binary_p = np.array([1, 0, 1, 0])
binary_q = np.array([1, 1, 0, 0])
hamming_distance = hamming(binary_p, binary_q) * len(binary_p)  # Multiply by length for actual count

# Print results
print(f"Euclidean Distance: {euclidean_distance}")
print(f"Manhattan Distance: {manhattan_distance}")
print(f"Cosine Similarity: {cosine_similarity}")
print(f"Minkowski Distance (p=3): {minkowski_distance}")
print(f"Hamming Distance: {hamming_distance}")
