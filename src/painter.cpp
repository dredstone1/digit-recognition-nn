#include "painter.hpp"

Painter::Painter() {
}

void Painter::open() {
}

void Painter::reset() {
}

App::App() {
}

void App::open() {
	if (running)
		return;

	displayThread = std::thread(&App::start, this);
	while (!painter) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	running = true;
}

void App::start() {
	painter = std::make_unique<Painter>();
}

void App::reset() {
	if (running) {
		painter->open();
	}
}

const std::vector<float> &App::getValues() {
	if (running) {
		return painter->getValues();
	}

	static const std::vector<float> empty; // fallback static empty vector
	return empty;
}
