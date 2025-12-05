#include "../include/painter.hpp"
#include "../include/transformation.hpp"
#include "consts.hpp"
#include <algorithm>
#include <iostream>
#include <model.hpp>

const int TOP_X = 5;

static App display;
static bool isOpen = false;

constexpr bool SHOW_LIVE_T = false;

void move(const nn::global::Tensor &p, nn::global::Tensor &result);

std::vector<nn::global::ValueType> temp(784);

std::vector<nn::global::ValueType> data(784);
static nn::global::Transformation doTransform = [](const nn::global::Tensor &p) {
	static nn::global::Tensor sample = p;

	if (SHOW_LIVE_T) {
		if (!isOpen)
			display.open();
		isOpen = true;
		sample.getData(data);
		display.setValues(data);
		std::this_thread::sleep_for(std::chrono::seconds(3)); // 3 seconds
	}

	if (nn::global::Tensor::getGpuState()) {
		move(p, sample);

	} else {
		tr::box newBox = tr::getBox(p);

		int v = tr::getAction(-newBox.x, 28 - (newBox.width + newBox.x));
		int h = tr::getAction(-newBox.y, 28 - (newBox.height + newBox.y));

		tr::move(sample, newBox, h, v);
	}

	if (SHOW_LIVE_T) {
		std::this_thread::sleep_for(std::chrono::seconds(3)); // 3 seconds
		sample.getData(data);
		display.setValues(data);
	}
	return sample;
};

int main(int argc, char *argv[]) {
	nn::global::Tensor::toGpu();
	nn::model::Model model("../ModelData/emnist_config.json");

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-l") {
			std::cout << "loading command emnist\n";
			model.load("emnist_model.txt");
		} else if (arg == "-t") {
			std::cout << "training command emnist\n";
			std::vector<std::string> files{
			    "../ModelData/Pemnist_balanced_train_data"};

			model.train(files, doTransform, doTransform);
			model.save("emnist_model.txt");
		}
	}

	std::vector<std::string> files{
	    "../ModelData/Pemnist_balanced_test_data"};
	nn::model::modelResult result = model.evaluateModel(files, doTransform);
	printf("final score evaluation: %f\n", result.percentage);
	if (!isOpen)
		display.open();
	isOpen = true;

	while (display.isOpen()) {
		display.wait();
		nn::global::Tensor metrix({784});
		metrix = display.getValues();

		model.runModel(metrix);
		std::vector<nn::global::ValueType> output = model.getOut();

		// Create vector of pairs (index, value)
		std::vector<std::pair<size_t, nn::global::ValueType>> indexedOutput;
		for (size_t i = 0; i < output.size(); ++i) {
			indexedOutput.push_back({i, output[i]});
		}

		// Sort descending by value
		std::sort(indexedOutput.begin(), indexedOutput.end(),
		          [](const auto &a, const auto &b) { return a.second > b.second; });

		// Print top X in one line
		for (int i = 0; i < std::min(TOP_X, (int)indexedOutput.size()); ++i) {
			size_t idx = indexedOutput[i].first;
			nn::global::ValueType val = indexedOutput[i].second;
			std::string character = emnist_balanced[idx].name;
			std::cout << character << ":" << val;
			if (i != std::min(TOP_X, (int)indexedOutput.size()) - 1)
				std::cout << " | ";
		}
		std::cout << std::endl;
	}

	return 0;
}
