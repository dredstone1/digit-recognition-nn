#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <tensor.hpp>

using namespace nn::global;

class RandomGenerator {
	std::mt19937 gen;

  public:
	RandomGenerator() : gen(std::random_device{}()) {}
	int getInt(int start, int end) {
		if (start > end)
			return start;
		std::uniform_int_distribution<> dist(start, end);
		return dist(gen);
	}

	std::mt19937 &getEngine() { return gen; }
};

static RandomGenerator rng;
int getAction(const int start, const int end) {
	return rng.getInt(start, end);
}

int getBiasedAction(int min_val, int max_val) {
	if (min_val == max_val)
		return min_val;

	// static constexpr float edge_bias_prob = 0.2f; // more bias toward edges
	float p = getAction(0, 10000) / 10000.0f;

	// if (p < edge_bias_prob) {
	//     // Pick left/top edge or right/bottom edge
	//     if (getAction(0, 1) == 0) {
	//         // Strong push toward min side
	//         return min_val;
	//     } else {
	//         // Strong push toward max side
	//         return max_val;
	//     }
	// } else {
	// Occasionally choose something in between
	return getAction(min_val, max_val);
	// }
}

constexpr int GRID_DIM = 28;
constexpr size_t SIZE = GRID_DIM * GRID_DIM; // 784

struct BoundingBox {
	int x_min = GRID_DIM;
	int y_min = GRID_DIM;
	int x_max = -1;
	int y_max = -1;
};

__global__ void moveKernel(const ValueType *original_data, ValueType *new_data, int horizontal_shift, int vertical_shift) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= SIZE)
		return;

	int dest_row = idx / GRID_DIM;
	int dest_col = idx % GRID_DIM;

	int source_row = dest_row - vertical_shift;
	int source_col = dest_col - horizontal_shift;

	if (source_row >= 0 && source_row < GRID_DIM &&
	    source_col >= 0 && source_col < GRID_DIM) {
		int source_idx = source_row * GRID_DIM + source_col;
		new_data[idx] = original_data[source_idx];
	}
}

// Simple device RNG for noise
__device__ float simpleHashRand(int seed, int idx) {
	unsigned int x = seed ^ idx;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return (x % 1000) / 1000.0f; // [0,1)
}

// Kernel to add noise in range [noise_range_low, noise_range_high]
__global__ void addNoiseKernel(ValueType *data, float noise_range_low, float noise_range_high, int seed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= SIZE)
		return;

	if (data[idx] > 0) {
		float noise = noise_range_low + (noise_range_high - noise_range_low) * simpleHashRand(seed, idx);
		float val = data[idx] + noise;

		val = fminf(fmaxf(val, 0.0f), 1.0f); // clamp to [0,1]

		data[idx] = val;
	}
}

// Updated move function
void move(const nn::global::Tensor &p, nn::global::Tensor &result) {
	std::vector<ValueType> host_data(SIZE);
	cudaMemcpy(host_data.data(), p.getGpuData(), SIZE * sizeof(ValueType), cudaMemcpyDeviceToHost);

	BoundingBox box;
	bool content_found = false;
	for (int i = 0; i < SIZE; ++i) {
		if (host_data[i] > 0.05) {
			content_found = true;
			int row = i / GRID_DIM;
			int col = i % GRID_DIM;
			box.x_min = std::min(box.x_min, col);
			box.x_max = std::max(box.x_max, col);
			box.y_min = std::min(box.y_min, row);
			box.y_max = std::max(box.y_max, row);
		}
	}

	if (!content_found) {
		cudaMemset(result.getGpuData(), 0, SIZE * sizeof(ValueType));
		return;
	}

	int h_shift_min = -box.x_min;
	int h_shift_max = (GRID_DIM - 1) - box.x_max;

	int v_shift_min = -box.y_min;
	int v_shift_max = (GRID_DIM - 1) - box.y_max;

	int final_h_shift = getBiasedAction(h_shift_min, h_shift_max);
	int final_v_shift = getBiasedAction(v_shift_min, v_shift_max);

	std::size_t blockSize = 256;
	std::size_t numBlocks = (SIZE + blockSize - 1) / blockSize;

	cudaMemset(result.getGpuData(), 0, SIZE * sizeof(ValueType));

	moveKernel<<<numBlocks, blockSize>>>(
	    p.getGpuData(),
	    result.getGpuData(),
	    final_h_shift,
	    final_v_shift);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error in moveKernel: %s\n", cudaGetErrorString(err));
	}

	cudaDeviceSynchronize();

	constexpr float noise_low = -0.1f;
	constexpr float noise_high = 0.1f;
	int noise_seed = final_h_shift * 1000 + final_v_shift;

	addNoiseKernel<<<numBlocks, blockSize>>>(
	    result.getGpuData(),
	    noise_low,
	    noise_high,
	    noise_seed);

	cudaError_t err2 = cudaGetLastError();
	if (err2 != cudaSuccess) {
		fprintf(stderr, "CUDA error in addNoiseKernel: %s\n", cudaGetErrorString(err2));
	}

	cudaDeviceSynchronize();
}
