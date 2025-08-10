#include "../include/painter.hpp"
#include "../include/transformation.hpp"
#include <iostream>
#include <model.hpp>

static App display;
static bool isOpen = false;

static nn::global::Transformation doTransform = [](const nn::global::Tensor &p) {
	nn::global::Tensor newSample = p;

	tr::stablize(newSample);

	// tr::box gridBox = tr::getBox(newSample);
	// tr::addMovement(newSample, gridBox, 3);
	return newSample;
};

static nn::global::Transformation finalEvaluate = [](const nn::global::Tensor &p) {
	nn::global::Tensor newSample = p;

	tr::stablize(newSample);

	// tr::box gridBox = tr::getBox(newSample);
	// tr::addMovement(newSample, gridBox);
	return newSample;
};

const int EMNIST_BALANCED_MAP[47] = {
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57,             // 0–9 → '0'-'9'
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74,             // 10–19 → 'A'-'J'
    75, 76, 77, 78, 79, 80, 81, 82, 83, 84,             // 20–29 → 'K'-'T'
    85, 86, 87, 88, 89, 90,                             // 30–35 → 'U'-'Z'
    97, 98, 100, 101, 102, 103, 104, 110, 113, 116, 117 // 36–46 → 'a','b','d','e','f','g','h','n','q','t','u'
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
			    "../ModelData/emnist_balanced_train_data"};

			model.train(files, doTransform, finalEvaluate);
			model.save("emnist_model.txt");
		}
	}

	nn::model::modelResult result = model.evaluateModel("../ModelData/emnist_balanced_test_data", finalEvaluate);
	printf("prediction: %f\n", result.percentage);
	if (!isOpen)
		display.open();
	isOpen = true;

	while (display.isOpen()) {
		display.wait();
		nn::global::Tensor metrix({784});
		metrix = display.getValues();

		model.runModel(metrix);
		nn::global::Prediction pre = model.getPrediction();

		char character = (pre.index < 47) ? static_cast<char>(EMNIST_BALANCED_MAP[pre.index]) : '?';

		printf("output: %zu, %f, char: %c\n", pre.index, pre.value, character);
	}

	return 0;
}
