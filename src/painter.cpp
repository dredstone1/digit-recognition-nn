#include "painter.hpp"

Painter::Painter()
    : window(sf::VideoMode({800, 600}), "SFML Window") {
}

void Painter::open() {
	while (window.isOpen()) {
		window.clear();

		window.display();
	}

	window.close();
}

void Painter::reset() {
}

void Painter::cancleEnter() {
	enter = false;
}

bool Painter::checkEnter() const {
	return enter;
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

}

void App::start() {
	if (painter)
		return;

	painter = std::make_unique<Painter>();
	running = true;
	painter->open();
}

void App::wait() {
	if (!running) {
		return;
	}

	while (!painter->checkEnter()) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	painter->cancleEnter();
}

const std::vector<float> &App::getValues() {
	if (running) {
		return painter->getValues();
	}

	static const std::vector<float> empty; // fallback static empty vector
	return empty;
}
