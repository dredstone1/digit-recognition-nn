#include "../include/transformation.hpp"

namespace tr {
int RandomGenerator::getInt(int start, int end) {
	std::uniform_int_distribution<> dist(start, end);
	return dist(gen);
}

static RandomGenerator rng;

int getAction(const int start, const int end) {
	return rng.getInt(start, end);
}

box getBox(const nn::global::ParamMetrix &metrix) {
	int min_x = 28, min_y = 28;
	int max_x = -1, max_y = -1;

	for (int i = 0; i < 784; ++i) {
		if (metrix[i] > 0) {
			int row = i / 28;
			int col = i % 28;
			min_x = std::min(min_x, col);
			max_x = std::max(max_x, col);
			min_y = std::min(min_y, row);
			max_y = std::max(max_y, row);
		}
	}

	if (max_x == -1 || max_y == -1) {
		return box{0, 0, 0, 0};
	}

	return box{
	    min_x,
	    min_y,
	    max_x - min_x + 1,
	    max_y - min_y + 1};
}

void clearOutsideBox(nn::global::ParamMetrix &metrix, const box &bound) {
	for (int y = 0; y < 28; ++y) {
		for (int x = 0; x < 28; ++x) {
			if (x < bound.x || x >= bound.x + bound.width ||
			    y < bound.y || y >= bound.y + bound.height) {
				metrix[y * 28 + x] = 0.0f;
			}
		}
	}
}

void shrinkBoxBound(box &boxData, const int a) {
	if (boxData.height > a * 2) {
		boxData.height -= a * 2;
		boxData.y += a;
	}

	if (boxData.width > a * 2) {
		boxData.width -= a * 2;
		boxData.x += a;
	}
}

void move(nn::global::ParamMetrix &metrix, const box &bound, const int h, const int v) {
	static thread_local nn::global::ParamMetrix temp(28 * 28);
	std::fill(temp.begin(), temp.end(), 0.0f);

	for (int y = 0; y < bound.height; ++y) {
		for (int x = 0; x < bound.width; ++x) {
			int src_x = bound.x + x;
			int src_y = bound.y + y;

			int dst_x = src_x + h;
			int dst_y = src_y + v;

			if (dst_x >= 0 && dst_x < 28 && dst_y >= 0 && dst_y < 28) {
				temp[dst_y * 28 + dst_x] = metrix[src_y * 28 + src_x];
			}
		}
	}

	std::swap(metrix, temp);
}

void addMovement(nn::global::ParamMetrix &metrix, box &gridBox, int shift) {
	int up3 = gridBox.y + shift;
	int down3 = 28 - (gridBox.y + gridBox.height) + shift;
	int left3 = gridBox.x + shift;
	int right3 = 28 - (gridBox.x + gridBox.width) + shift;

	int horizontal = getAction(-left3, right3);
	int vertical = getAction(-up3, down3);

	if (horizontal > 0) {
		int rightEdgeAfterMove = gridBox.x + gridBox.width + horizontal;
		if (rightEdgeAfterMove > 28) {
			gridBox.width -= (rightEdgeAfterMove - 28);
		}
	}
	if (vertical > 0) {
		int bottomEdgeAfterMove = gridBox.y + gridBox.height + vertical;
		if (bottomEdgeAfterMove > 28) {
			gridBox.height -= (bottomEdgeAfterMove - 28);
		}
	}
	if (horizontal < 0) {
		int leftEdgeAfterMove = gridBox.x + horizontal;
		if (leftEdgeAfterMove < 0) {
			int overflow = -leftEdgeAfterMove;
			gridBox.x += overflow;
			gridBox.width -= overflow;
		}
	}
	if (vertical < 0) {
		int topEdgeAfterMove = gridBox.y + vertical;
		if (topEdgeAfterMove < 0) {
			int overflow = -topEdgeAfterMove;
			gridBox.y += overflow;
			gridBox.height -= overflow;
		}
	}

	clearOutsideBox(metrix, gridBox);

	move(metrix, gridBox, horizontal, vertical);
}
void stablize(nn::global::ParamMetrix &metrix) {
	for (auto &value : metrix) {
		value /= 255;
	}
}
} // namespace tr
