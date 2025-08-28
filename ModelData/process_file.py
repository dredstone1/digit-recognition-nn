def process_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        # Read and copy the first line as-is
        first_line = infile.readline()
        outfile.write(first_line)

        for line in infile:
            parts = line.strip().split()
            if len(parts) != 785:
                # Skip lines that don't have exactly 1 label + 784 values
                continue
            label = parts[0]  # Keep the first number unchanged
            pixels = parts[1:]  # The 784 pixel values
            normalized = [str(float(p) / 255.0) for p in pixels]
            outfile.write('p' + label + ' ' + ' '.join(normalized) + '\n')

if __name__ == "__main__":
    input_file = 'emnist_balanced_train_data.nndb'
    output_file = 'Pemnist_balanced_train_data.nndb'
    process_file(input_file, output_file)

    input_file = 'emnist_balanced_test_data.nndb'
    output_file = 'Pemnist_balanced_test_data.nndb'
    process_file(input_file, output_file)

    input_file = 'train_data.nndb'
    output_file = 'Pemnist_train_data.nndb'
    process_file(input_file, output_file)
