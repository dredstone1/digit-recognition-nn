#include "../include/painter.hpp"
#include "Globals.hpp"
#include "../include/transformation.hpp"
#include <iostream>
#include <model.hpp>

static nn::global::Transformation doTransform = [](const nn::global::ParamMetrix &p) {
	static App display;
	static bool isOpen = false;
	if (!isOpen)
		display.open();
	isOpen = true;

	nn::global::ParamMetrix newSample = p;

	tr::stablize(newSample);

	// Apply movement
	tr::box gridBox = tr::getBox(newSample);
	tr::addMovment(newSample, gridBox);

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

	tr::stablize(newSample);

	// Apply movement
	tr::box gridBox = tr::getBox(newSample);
	tr::addMovment(newSample, gridBox);

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
			std::cout << "loading command emnist\n";
			model.load("emnist_model.txt");
		} else if (arg == "-t") {
			std::cout << "training command emnist\n";
			model.train("../ModelData/emnist_train_data", doTransform);
			model.save("emnist_model.txt");
		}
	}

	nn::model::modelResult result = model.evaluateModel("../ModelData/emnist_test_data", finalEvaluate);
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
