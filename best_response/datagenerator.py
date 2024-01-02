import random

# Generate 10 random points within [0, 1] on both x and y axes
points = [(random.uniform(0, 100), 0) for _ in range(50)]

# Store the coordinates into a file
file_path = 'best_response/data2.txt'
with open(file_path, 'w') as file:
    for point in points:
        file.write(f"{point[0]:.6f} {point[1]:.6f}\n")

print(f"Random points have been stored in {file_path}")