#include <bitset>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cinput.h"

extern std::vector<Node> nodes;
extern std::vector<std::vector<int>> edges;
extern std::vector<int> starts;

std::bitset<65> visited;
std::bitset<65> starts_mask;
std::vector<std::pair<int, int>> stack;

std::bitset<65> cur_starts_mask;
std::bitset<65> ground_starts_mask;

void traverse(int start, int w, int target) {
  if (w > target) return;
  if (w == target) {
    int redge = rotation_edges[start];
    if (redge == -1) return;

    if (target == 35 * 4) {
      // std::cerr << "found one\n";
      // std::cerr << "[";
      // for (auto [id, r]: stack) {
      //   std::cerr << "("<< id << ", " << r << "), ";
      // }
      // std::cerr << "]\n" << w << "\n";

      if (cur_starts_mask != ground_starts_mask) {
        std::cerr << cur_starts_mask << "\n";
        ground_starts_mask = cur_starts_mask;
      }
      return;
    }

    // std::cerr << "rotation!\n";
    const auto& node = nodes[redge];
    if (!starts_mask[node.id]) {
      std::cerr << "its not a start lol\n";
    }
    // std::cerr << stack.back() << ", " << node.id << "\n";
    // if (stack.back() != node.id) {
    //   exit(0);
    // }
    stack.emplace_back(node.id, node.r);
    cur_starts_mask[node.id] = 1;
    traverse(redge, w + node.w, target + 35);
    cur_starts_mask[node.id] = 0;
    stack.pop_back();
    return;
  }

  for (const auto n : edges[start]) {
    const auto& node = nodes[n];
    if (visited[node.id]) continue;
    visited[node.id] = 1;
    stack.emplace_back(node.id, node.r);
    traverse(n, w + node.w, target);
    stack.pop_back();
    visited[node.id] = 0;
  }
}

int main(int, char**) {
  std::vector<int> back_rotation_edges(rotation_edges.size(), -1);
  for (int i = 0; i < rotation_edges.size(); ++i) {
    back_rotation_edges[rotation_edges[i]] = i;
  }

  for (const auto start : starts) {
    starts_mask[nodes[start].id] = 1;
  }

  int target = 35;
  // int target = 35*4;
  int start_id = rand() % starts.size();
  for (auto start : starts) {
    start = starts[start_id];
    const auto bre = back_rotation_edges[start];
    const auto& node = nodes[start];
    std::cerr << start << "\n";
    visited[node.id] = 1;
    stack.emplace_back(node.id, node.r);
    traverse(start, node.w + nodes[bre].w, target);
    stack.pop_back();
    visited[node.id] = 0;
  }
}
