#include "../include/painter.hpp"
#include "Globals.hpp"
#include <AiModel.hpp>

int getAction(const int start, const int end) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dist(start, end);

	int number = dist(gen);
	return number;
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

	for (int row = 0; row < 28; ++row) {
		for (int col = 0; col < 28; ++col) {
			if (metrix[row * 28 + col] > 0) {
				min_x = std::min(min_x, col);
				max_x = std::max(max_x, col);
				min_y = std::min(min_y, row);
				max_y = std::max(max_y, row);
			}
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
	const int size = 28;
	nn::global::ParamMetrix temp(size * size, 0.0f);

	for (int y = 0; y < bound.height; ++y) {
		for (int x = 0; x < bound.width; ++x) {
			int src_x = bound.x + x;
			int src_y = bound.y + y;

			int dst_x = src_x + h;
			int dst_y = src_y + v;

			if (dst_x >= 0 && dst_x < size && dst_y >= 0 && dst_y < size) {
				float value = metrix[src_y * size + src_x];
				temp[dst_y * size + dst_x] = value;
			}
		}
	}

	metrix = std::move(temp);
}

nn::global::Transformation doTransform = [](const nn::global::ParamMetrix &p) {
	nn::global::ParamMetrix newSample = p;

	box gridBox = getBox(newSample);

	int up = gridBox.y;
	int down = 28 - (gridBox.y + gridBox.height);
	int left = gridBox.x;
	int right = 28 - (gridBox.x + gridBox.width);

	int horizotal = getAction(-left, right);
	int vertical = getAction(-up, down);

	move(newSample, gridBox, horizotal, vertical);

	return newSample;
};

int main() {
	nn::AiModel model("../ModelData/config.json");

	model.load("params");
	model.train("../ModelData/data1", doTransform);
	model.save("params");

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
