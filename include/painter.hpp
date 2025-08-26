#ifndef PAINTER
#define PAINTER

#include <SFML/Graphics.hpp>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

constexpr sf::Color BG_COLOR(100, 100, 100);
constexpr sf::Color PANELS_BG(255, 255, 255);

constexpr std::uint32_t WINDOW_HEIGHT = 600;
constexpr std::uint32_t WINDOW_WIDTH = 600;

constexpr std::uint32_t UI_GAP = 15;
constexpr float PIXEL_GAP = 1;

constexpr std::uint32_t CANVAS_SIZE = WINDOW_HEIGHT - UI_GAP * 2;

constexpr int GRID_SIZE = 28;

constexpr float CELL_SIZE = (CANVAS_SIZE - (GRID_SIZE - 1) * PIXEL_GAP) / GRID_SIZE;

const std::string WINDOW_NAME = "painter";

constexpr int DEFAULT_BRUSH_RADIUS = 2;

enum class MouseMode {
	paint,
	remove,
	none,
};

class Painter {
  private:
	sf::RenderWindow window;
	std::atomic<bool> enter{false};
	std::atomic<bool> running{true};

	std::vector<float> values;

	int brushRadius = DEFAULT_BRUSH_RADIUS;

    bool windowResize{false};

	MouseMode mouseActive{MouseMode::none};

	void reset();

	void processEvents();
	void renderCanvas();

	void drawCanvas();
	void applyBrush();

    void resetSize();

  public:
	Painter();
	~Painter() = default;

	void open();
	void close();

	bool checkEnter() const;
	void cancleEnter();

	const std::vector<float> &getValues() { return values; }
	void setValues(const std::vector<float> &values_) { values = values_; }

	bool isRunning() { return running; }
};

class App {
  private:
	std::thread displayThread;
	std::unique_ptr<Painter> painter;
	std::atomic<bool> running{false};

	void start();
	void close();

  public:
	App() {}
	~App() = default;

	void open();
	void wait();

	bool isOpen() { return running; }

	const std::vector<float> &getValues();
	void setValues(const std::vector<float> &values) { painter->setValues(values); }
};

#endif // PAINTER
