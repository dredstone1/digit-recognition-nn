#include "../include/painter.hpp"
#include <algorithm>
#include <cmath>

Painter::Painter()
    : window(sf::VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}), WINDOW_NAME),
      values(GRID_SIZE * GRID_SIZE) {
}

void Painter::open() {
	while (window.isOpen() && running) {
		window.clear(BG_COLOR);

		processEvents();
		renderCanvas();

		window.display();
	}

	window.close();
}

void Painter::close() {
	running = false;
}

void Painter::processEvents() {
	while (const std::optional event = window.pollEvent()) {
		if (event->is<sf::Event::Closed>()) {
			running = false;
		} else if (const auto *mouseButtonPressed = event->getIf<sf::Event::MouseButtonPressed>()) {
			if (mouseButtonPressed->button == sf::Mouse::Button::Right) {
				mouseActive = MouseMode::remove;
			} else {
				mouseActive = MouseMode::paint;
			}
		} else if (event->is<sf::Event::MouseButtonReleased>()) {
			mouseActive = MouseMode::none;
		} else if (const auto *keyPressed = event->getIf<sf::Event::KeyPressed>()) {
			if (keyPressed->scancode == sf::Keyboard::Scancode::Enter) {
				enter = true;
			} else if (keyPressed->scancode == sf::Keyboard::Scancode::R) {
				reset();
			} else if (keyPressed->scancode == sf::Keyboard::Scancode::Down) {
				if (brushRadius > 0)
					brushRadius--;
			} else if (keyPressed->scancode == sf::Keyboard::Scancode::Up) {
				if (brushRadius < 10)
					brushRadius++;
			}
		}
	}
}

void Painter::renderCanvas() {
	if (mouseActive != MouseMode::none) {
		applyBrush();
	}
	drawCanvas();
}

void Painter::applyBrush() {
	sf::Vector2i mousePos = sf::Mouse::getPosition(window);
	int mouseGridX = (mousePos.x - UI_GAP) / (CELL_SIZE + PIXEL_GAP);
	int mouseGridY = (mousePos.y - UI_GAP) / (CELL_SIZE + PIXEL_GAP);

	const float sigma = brushRadius * 0.5f;
	const float maxStrength = 25.f;
	const float radiusSquared = brushRadius * brushRadius;

	for (int dy = -brushRadius; dy <= brushRadius; ++dy) {
		for (int dx = -brushRadius; dx <= brushRadius; ++dx) {
			int px = mouseGridX + dx;
			int py = mouseGridY + dy;

			if (px >= 0 && px < GRID_SIZE && py >= 0 && py < GRID_SIZE) {
				float distSquared = dx * dx + dy * dy;
				if (distSquared <= radiusSquared) {
					float strength = std::exp(-distSquared / (2 * sigma * sigma));
					int index = py * GRID_SIZE + px;

					if (mouseActive == MouseMode::paint) {
						values[index] = std::min(255.f, values[index] + strength * maxStrength) / 255;
					} else {
						values[index] = std::max(0.f, values[index] - strength * maxStrength) / 255;
					}
				}
			}
		}
	}
}

void Painter::drawCanvas() {
	sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));
	int index = 0;

	for (int y = 0; y < GRID_SIZE; ++y) {
		for (int x = 0; x < GRID_SIZE; ++x) {
			cell.setPosition({UI_GAP + x * (CELL_SIZE + PIXEL_GAP),
			                  UI_GAP + y * (CELL_SIZE + PIXEL_GAP)});

			float value = std::clamp(values[index] * 255, 0.f, 255.f);
			cell.setFillColor(sf::Color(value, value, value));
			window.draw(cell);
			++index;
		}
	}
}

void Painter::reset() {
	for (auto &value : values) {
		value = 0;
	}
}

void Painter::cancleEnter() {
	enter = false;
}

bool Painter::checkEnter() const {
	return enter;
}

void App::open() {
	if (running)
		return;

	displayThread = std::thread(&App::start, this);
	while (!running) {
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

void App::close() {
	running = false;

	if (displayThread.joinable()) {
		if (painter) {
			painter->close();
		}

		displayThread.join();
	}
}

void App::wait() {
	if (!running) {
		return;
	}

	while (!painter->checkEnter() && running) {
		if (!painter->isRunning()) {
			close();
		}
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	painter->cancleEnter();
}

const std::vector<float> &App::getValues() {
	if (running) {
		return painter->getValues();
	}

	static const std::vector<float> empty;
	return empty;
}
