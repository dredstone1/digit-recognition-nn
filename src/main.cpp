#include "../include/painter.hpp"
#include "Globals.hpp"
#include <cstddef>
#include <iostream>
#include <model.hpp>

class RandomGenerator {
	std::mt19937 gen;

  public:
	RandomGenerator() : gen(std::random_device{}()) {}

	int getInt(int start, int end) {
		std::uniform_int_distribution<> dist(start, end);
		return dist(gen);
	}
};

static RandomGenerator rng;

static int getAction(const int start, const int end) {
	return rng.getInt(start, end);
}

struct box {
	int x;
	int y;
	int width;
	int height;
};

static box getBox(const nn::global::ParamMetrix &metrix) {
	int min_x = 28, min_y = 28;
	int max_x = -1, max_y = -1;

	for (int i = 0; i < 784; ++i) {
		if (metrix[i] > 0) {
			int row = i / 28;
			int col = i % 28;
			min_x = std::min(min_x, col);
			max_x = std::max(max_x, col);
			min_y = std::min(min_y, row);
			max_y = std::max(max_y, row);
		}
	}

	if (max_x == -1 || max_y == -1) {
		return box{0, 0, 0, 0};
	}

	return box{
	    min_x,
	    min_y,
	    max_x - min_x + 1,
	    max_y - min_y + 1};
}

static void move(nn::global::ParamMetrix &metrix, const box &bound, const int h, const int v) {
	static thread_local nn::global::ParamMetrix temp(28 * 28);
	std::fill(temp.begin(), temp.end(), 0.0f);

	for (int y = 0; y < bound.height; ++y) {
		for (int x = 0; x < bound.width; ++x) {
			int src_x = bound.x + x;
			int src_y = bound.y + y;

			int dst_x = src_x + h;
			int dst_y = src_y + v;

			if (dst_x >= 0 && dst_x < 28 && dst_y >= 0 && dst_y < 28) {
				temp[dst_y * 28 + dst_x] = metrix[src_y * 28 + src_x];
			}
		}
	}

	std::swap(metrix, temp);
}

static void addMovment(nn::global::ParamMetrix &metrix, const box &gridBox) {
	int up = gridBox.y;
	int down = 28 - (gridBox.y + gridBox.height);
	int left = gridBox.x;
	int right = 28 - (gridBox.x + gridBox.width);

	int horizotal = getAction(-left, right);
	int vertical = getAction(-up, down);

	move(metrix, gridBox, horizotal, vertical);
}

constexpr float noiseStrength = 255.0f;  // Max noise offset
constexpr float noiseProbability = 0.1f; // 5% chance to add noise

static void addNoise(nn::global::ParamMetrix &metrix) {
	static thread_local std::mt19937 gen(std::random_device{}());
	std::uniform_real_distribution<float> noiseDist(-noiseStrength, noiseStrength);
	std::uniform_real_distribution<float> chanceDist(0.0f, 1.0f);

	for (float &val : metrix) {
		if (chanceDist(gen) < noiseProbability) {
			val += noiseDist(gen);
			val = std::clamp(val, 1.0f, 255.0f); // Keep valid range
		}
	}
}

static void invert(nn::global::ParamMetrix &metrix) {
	for (size_t i = 0; i < metrix.size(); ++i) {
		metrix[i] = 255 - metrix[i];
	}
}

static nn::global::Transformation doTransform = [](const nn::global::ParamMetrix &p) {
	static App display;
	static bool isOpen = false;
	if (!isOpen)
		display.open();
	isOpen = true;

	nn::global::ParamMetrix newSample = p;

	// Apply movement
	box gridBox = getBox(newSample);
	addMovment(newSample, gridBox);

	// Apply noise
	if (rng.getInt(0, 1) == 0) {
		addNoise(newSample);
	}

	// Apply noise
	if (rng.getInt(0, 1) == 0) {
        invert(newSample);
	}

	display.setValues(newSample);
	return display.getValues();
};

int main(int argc, char *argv[]) {
	nn::model::Model model("../ModelData/config.json");

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-l") {
			std::cout << "loading command\n";
			model.load("model.txt");
		} else if (arg == "-t") {
			std::cout << "training command\n";
			model.train("../ModelData/data1", doTransform);
			model.save("model.txt");
		}
	}

	nn::model::modelResult result = model.evaluateModel("../ModelData/data");
	printf("prediction: %f\n", result.percentage);

	App display;
	display.open();

	while (display.isOpen()) {
		display.wait();

		model.runModel(display.getValues());
		printf("output: %zu, %f\n", model.getPrediction().index, model.getPrediction().value);
	}

	return 0;
}
