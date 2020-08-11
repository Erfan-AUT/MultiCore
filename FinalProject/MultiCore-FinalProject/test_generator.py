import os
import random

matrix_sizes = [4, 8, 11, 13, 32, 64]

current = os.getcwd()
new_dir = os.path.join(current, "data_in")
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

file_count = random.randint(1, 30)
for i in range(file_count):
    file_path = os.path.join(new_dir, str(i) + ".txt")
    with open(file_path, 'w') as file:
        sample_count = random.randint(1, 30)
        for j in range(sample_count):
            matrix_size = random.choice(matrix_sizes)
            sample_str = ""
            for k in range(matrix_size ** 2):
                element = random.uniform(0.0, 10.0)
                sample_str += str(element) + " "
            sample_str = sample_str.rstrip()
            file.write(sample_str + "\n\n")

