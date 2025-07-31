#include "Globals.hpp"
#include <random>

namespace tr {
class RandomGenerator {
	std::mt19937 gen;

  public:
	RandomGenerator() : gen(std::random_device{}()) {}

	int getInt(int start, int end);
};

int getAction(const int start, const int end);

struct box {
	int x;
	int y;
	int width;
	int height;
};

box getBox(const nn::global::ParamMetrix &metrix);
void shrinkBoxBound(box &boxData, const int a);

void clearOutsideBox(nn::global::ParamMetrix &metrix, const box &bound);

void move(nn::global::ParamMetrix &metrix, const box &bound, const int h, const int v);

void addMovement(nn::global::ParamMetrix &metrix,  box &gridBox, int shift = 0);
void stablize(nn::global::ParamMetrix &metrix);

} // namespace tr
