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

static void addNoise(nn::global::ParamMetrix &metrix) {
	static thread_local std::mt19937 gen(std::random_device{}());
	std::uniform_real_distribution<float> chanceDist(0.0f, 1.0f);
	std::uniform_real_distribution<float> noiseDist(0.0f, 1.0f);

	const float noiseLevel = 0.2f;

	for (float &val : metrix) {
		if (chanceDist(gen) < noiseLevel) {
			val += noiseDist(gen);
			val = std::min(val, 1.0f);
		}
	}
}

static void invert(nn::global::ParamMetrix &metrix) {
	for (size_t i = 0; i < metrix.size(); ++i) {
		metrix[i] = 1 - metrix[i];
	}
}

static void thinWidth(nn::global::ParamMetrix &metrix) {
	static thread_local nn::global::ParamMetrix temp(28 * 28, 0.0f);

	// Factor: how much to thin (1.0 = no change, <1 = thinner)
	const float shrinkFactor = 0.7f;
	const int newWidth = std::max(1, static_cast<int>(28 * shrinkFactor));

	// Clear temp
	std::fill(temp.begin(), temp.end(), 0.0f);

	for (int row = 0; row < 28; ++row) {
		for (int col = 0; col < newWidth; ++col) {
			// Map compressed col to original space
			float origXf = (float)col / newWidth * 28.0f;
			int x0 = static_cast<int>(origXf);
			int x1 = std::min(x0 + 1, 27);
			float weight = origXf - x0;

			float pixel = (1.0f - weight) * metrix[row * 28 + x0] +
			              weight * metrix[row * 28 + x1];

			// Recenter compressed image on original canvas
			int targetX = col + (28 - newWidth) / 2;
			temp[row * 28 + targetX] = pixel;
		}
	}

	std::swap(metrix, temp);
}

static void dimOpacity(nn::global::ParamMetrix &metrix) {
	float alpha = 0.5f + (rng.getInt(0, 20) / 40.0f);
	for (float &val : metrix) {
		val *= alpha;
	}
}

static void stablize(nn::global::ParamMetrix &metrix) {
	for (auto &value : metrix) {
		value /= 255;
	}
}

static nn::global::Transformation doTransform = [](const nn::global::ParamMetrix &p) {
	static App display;
	static bool isOpen = false;
	if (!isOpen)
		display.open();
	isOpen = true;

	nn::global::ParamMetrix newSample = p;

	stablize(newSample);

	// Apply thinning
	// if (rng.getInt(0, 5) == 0) {
	// 	thinWidth(newSample);
	// }

	// if (rng.getInt(0, 2) == 0) {
	// 	dimOpacity(newSample);
	// }

	// Apply movement
	box gridBox = getBox(newSample);
	addMovment(newSample, gridBox);

	// Apply noise
	// int noiseLevel = rng.getInt(0, 1);
	// if (noiseLevel == 0) {
	// 	addNoise(newSample);
	// }

	// Apply invert
	// if (rng.getInt(0, 1) == 0) {
	// 	invert(newSample);
	// }

	display.setValues(newSample);
	return display.getValues();
};

static nn::global::Transformation finalEvaluate = [](const nn::global::ParamMetrix &p) {
	static App display;
	static bool isOpen = false;
	if (!isOpen)
		display.open();
	isOpen = true;

	nn::global::ParamMetrix newSample = p;

	stablize(newSample);

	// Apply movement
	box gridBox = getBox(newSample);
	addMovment(newSample, gridBox);

	display.setValues(newSample);
	return display.getValues();
};

const int EMNIST_BALANCED_MAP[62] = {
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,     // 0–9 → '0'–'9'
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74,     // 10–19 → 'A'–'J'
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84,     // 20–29 → 'K'–'T'
    85, 86, 87, 88, 89, 90,                     // 30–35 → 'U'–'Z'
    97, 98, 99, 100, 101, 102, 103, 104, 105,   // 36–44 → 'a'–'i'
    106, 107, 108, 109, 110, 111, 112, 113,     // 45–52 → 'j'–'q'
    114, 115, 116, 117, 118, 119, 120, 121, 122 // 53–61 → 'r'–'z'
};

int main(int argc, char *argv[]) {
	nn::model::Model model("../ModelData/emnist_config.json");

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-l") {
			std::cout << "loading command\n";
			model.load("model.txt");
		} else if (arg == "-t") {
			std::cout << "training command\n";
			model.train("../ModelData/data1", doTransform);
			model.save("model.txt");
		} else if (arg == "-L") {
			std::cout << "loading command emnist\n";
			model.load("emnist_model.txt");
		} else if (arg == "-T") {
			std::cout << "training command emnist\n";
			model.train("../ModelData/emnist_train_data", doTransform);
			model.save("emnist_model.txt");
		}
	}

	nn::model::modelResult result = model.evaluateModel("../ModelData/emnist_test_data.nndb", finalEvaluate);
	printf("prediction: %f\n", result.percentage);

	App display;
	display.open();

	while (display.isOpen()) {
		display.wait();
		model.runModel(display.getValues());
        nn::global::Prediction pre = model.getPrediction();

		char character = (pre.index < 62) ? static_cast<char>(EMNIST_BALANCED_MAP[pre.index]) : '?';

		printf("output: %zu, %f, char: %c\n", pre.index, pre.value, character);
	}

	return 0;
}
