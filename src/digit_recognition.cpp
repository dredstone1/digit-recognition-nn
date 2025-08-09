#include "../include/painter.hpp"
#include "../include/transformation.hpp"
#include "tensor.hpp"
#include <Globals.hpp>
#include <iostream>
#include <model.hpp>
#include <ostream>

static nn::global::Transformation doTransform = [](const nn::global::Tensor &p) {
	static App display;
	static bool isOpen = false;
	if (!isOpen)
		display.open();
	isOpen = true;

	nn::global::Tensor newSample = p;

	tr::stablize(newSample);

	// Apply movement
	tr::box gridBox = tr::getBox(newSample);
	addMovement(newSample, gridBox);

	// display.setValues(newSample.getData());
	return newSample;
};

static nn::global::Transformation finalEvaluate = [](const nn::global::Tensor &p) {
	static App display;
	static bool isOpen = false;
	if (!isOpen)
		display.open();
	isOpen = true;

	nn::global::Tensor newSample = p;

	tr::stablize(newSample);

	// Apply movement
	tr::box gridBox = tr::getBox(newSample);
	addMovement(newSample, gridBox);

	// display.setValues(newSample.getData());
	return newSample;
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

	nn::model::modelResult result = model.evaluateModel("../ModelData/test_data", finalEvaluate);
	std::cout << "evaluation: " << result.percentage << std::endl;

	App display;
	display.open();

	while (display.isOpen()) {
		display.wait();
		nn::global::Tensor metrix({784});
        metrix = display.getValues();

		model.runModel(metrix);
		nn::global::Prediction pre = model.getPrediction();

		std::cout << "prediction: " << pre.index << ", " << pre.value << std::endl;
	}

	return 0;
}
