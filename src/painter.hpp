#ifndef PAINTER
#define PAINTER

#include <SFML/Graphics.hpp>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

class Painter {
  private:
	std::vector<float> values;

  public:
	Painter();
	~Painter() = default;

	void open();
	void reset();
	const std::vector<float> &getValues() { return values; }
};

class App {
  private:
	std::thread displayThread;
	std::unique_ptr<Painter> painter;
	std::atomic<bool> running{false};

    void start();

  public:
	App();
	~App() = default;

	void open();
	void reset();
	void wait();
	const std::vector<float> &getValues();
};

#endif // PAINTER
