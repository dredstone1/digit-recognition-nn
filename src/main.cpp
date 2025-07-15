#include "../include/painter.hpp"
#include <AiModel.hpp>

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

int getAction(const int start, const int end) {
	return rng.getInt(start, end);
}

struct box {
	int x;
	int y;
	int width;
	int height;
};

box getBox(const nn::global::ParamMetrix &metrix) {
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

void move(nn::global::ParamMetrix &metrix, const box &bound, const int h, const int v) {
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

nn::global::Transformation doTransform = [](const nn::global::ParamMetrix &p) {
	static App display;
	static bool isOpen = false;
	if (!isOpen)
		display.open();
	isOpen = true;
	// display.setValues(p);

	nn::global::ParamMetrix newSample = p;

	box gridBox = getBox(newSample);

	int up = gridBox.y;
	int down = 28 - (gridBox.y + gridBox.height);
	int left = gridBox.x;
	int right = 28 - (gridBox.x + gridBox.width);

	int horizotal = getAction(-left, right);
	int vertical = getAction(-up, down);

	move(newSample, gridBox, horizotal, vertical);

	display.setValues(newSample);
	return display.getValues();
};

int main(int argc, char *argv[]) {
	nn::AiModel model("../ModelData/config.json");

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-l") {
			model.load("params");
		} else if (arg == "-t") {
			model.train("../ModelData/data1", doTransform);
			model.save("params");
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
