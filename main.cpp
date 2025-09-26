#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Grid {
  void print(std::ostream& os) const {
    bool even = true;
    int rowid = 0;
    printf("  :");
    for (int i = 0; i < W; ++i) {
      printf("%2d", i);
    }
    printf("\n");
    for (size_t row : field) {
      printf("%2d:", rowid++);
      for (size_t cell = 0; cell < W+10; ++cell) {
        if (row & (1ull << cell)) {
          if (even ^ (cell & 1)) {
            os << "()";
          } else {
            os << (even ? "><" : "><");
          }
        } else {
          os << "\033[38;5;245m .\033[0m";
        }
      }
      os << "\n";
      even ^= true;
    }
    os << "\n";
  }

  std::vector<size_t> field;
  size_t W = 0;
};

struct Detail : public Grid {
  Detail(std::string_view label, std::vector<std::string> rows)
      : label(label), rows(std::move(rows)) {
    compile();
    // print(std::cerr);
  }

  void compile() {
    field.clear();
    for (const auto& row : rows) {
      W = std::max(W, row.size() >> 1);
      size_t chunk = 0;
      for (auto it = row.rbegin(); it < row.rend(); it += 2) {
        chunk = (chunk << 1) | (*it != ' ');
      }
      field.push_back(chunk);
    }
  }

  void rotate() {
    for (int i = 0; i < rows.size(); ++i) {
      for (int j = i; j < rows.size(); ++j) {
        std::swap(rows[i][2 * j], rows[j][2 * i]);
        std::swap(rows[i][2 * j + 1], rows[j][2 * i + 1]);
      }
    }
    for (int i = 0; i < rows.size(); ++i) {
      for (int j = 0; j < rows.size() / 2; ++j) {
        int mirrorj = rows.size() - 1 - j;
        std::swap(rows[i][2 * j], rows[i][2 * mirrorj]);
        std::swap(rows[i][2 * j + 1], rows[i][2 * mirrorj + 1]);
      }
    }
    compile();
  }

  void print(std::ostream& os) const {
    os << label << "\n";
    Grid::print(os);
  }

  std::string label;

 private:
  std::vector<std::string> rows;
  friend class Field;
};

class Field : public Grid {
 public:
  static constexpr size_t W = 35;
  static constexpr size_t H = 35;

  Field(std::vector<Detail> details) : details(std::move(details)) {
    Grid::W = W;
    field.resize(H, ~((1ull << W) - 1));
    mask.W = W;
    mask.field.resize(H, 0);
  }

  struct Item {
    int id;
    int x;
    int y;
  };

  bool put(int id, int x, int y) {
    if (id > details.size()) return false;
    const auto detail = details[id];
    if (x + detail.field.size() > field.size()) return false;
    const int base_x = x;
    for (const auto& row : detail.field) {
      if ((row << y) & field[x++]) return false;
    }
    x = base_x;
    for (const auto& row : detail.field) {
      field[x++] |= (row << y);
    }
    stack.push_back(Item{id, base_x, y});
    return true;
  }

  void pop() {
    if (stack.empty()) return;
    auto last_item = stack.back();
    for (const auto& row : details[last_item.id].field) {
      field[last_item.x++] &= ~(row << last_item.y);
    }
  }
  void checksen() {
    auto checksen = [this](auto self, int d) -> void {
      if (d >= details.size()) {
        print(std::cerr);
        std::cerr << "solution:\n";
        for (const auto item : stack) {
          std::cerr << item.id << " " << item.x << " " << item.y;
        std::cerr << "\n";
        }
        // exit(0);
        return;
      }
      for (int i = 0; i < 4; ++i) {
        for (int x = 0; x < Field::H; ++x) {
          for (int y = 0; y < Field::W; ++y) {
            if ((x & 1) ^ (y & 1)) continue;
            if (!put(d, x, y)) continue;
            self(self, d + 1);
            pop();
          }
        }
        details[d].rotate();
      }
    };
    for (int d = 0; d < details.size(); ++d) {
      checksen(checksen, d);
    }
  }

  auto match() {
    std::vector<size_t> available_ids;
    for (size_t i = 0; i < details.size(); ++i)
      if (std::find_if(stack.begin(), stack.end(),
                       [i](auto item) { return item.id == i; }) == stack.end())
        available_ids.push_back(i);

    for (auto id : available_ids) {
      auto& detail = details[id];
      for (int i = 0; i < 4; ++i) {
        for (int x = 0; x < Field::H; ++x) {
          for (int y = 0; y < Field::W; ++y) {
            if ((x & 1) ^ (y & 1)) continue;
            if (!put(id, x, y)) continue;
            if (check_mask()) {
              std::cerr << "matched with: " << detail.label << "\n";
              print(std::cerr);
            }
            pop();
          }
        }

        detail.rotate();
      }
    }
  };

  auto putmask(int x, int y) { mask.field[x] |= (1 << y); }
  auto clrmask(int x, int y) { mask.field[x] &= ~(1 << y); }
  auto showmask(std::ostream& os) const {
    os << "mask:\n";
    mask.print(os);
  }

  bool check_mask() const {
    auto correct = true;
    for (int x = 0; correct && x < Field::H; ++x) {
      correct = (field[x] & mask.field[x]) == mask.field[x];
    }
    return correct;
  }

  auto rotate() {
    if (stack.empty()) return;
    auto last_item = stack.back();
    pop();
    do {
      details[last_item.id].rotate();
    } while (!put(last_item.id, last_item.x, last_item.y));
  }

 private:
  std::vector<Detail> details;
  std::vector<Item> stack;

  Grid mask;
};

// {
//   "()==()==()",
//   "||  ||  ||",
//   "()  ()  ()",
// }
std::vector<Detail> details = {
#include "details2.h"
};

void column_printer(std::vector<Detail>& details) {
  std::ostringstream ss;
  size_t max_field_size = 0;
  for (const auto& detail : details) {
    max_field_size = std::max(detail.field.size(), max_field_size);
  }
  for (const auto& detail : details) {
    detail.print(ss);
    ss << std::string(max_field_size - detail.field.size(), '\n');
  }
}

int main(int, char**) {
  Field field(details);
  // field.checksen();
  std::unordered_map<std::string, size_t> field_ids;
  for (size_t i = 0; i < details.size(); ++i) field_ids[details[i].label] = i;

  std::string cmd;

  int x = Field::H / 2 - 5;
  int y = Field::H / 2 - 5;

  field.print(std::cerr);
  while (std::cin >> cmd) {
    if (cmd.empty()) continue;
    std::cerr << "\033[2J";
    if (cmd.front() == '.') {
      if (!field_ids.count(cmd)) {
        std::cerr << "no such id\n";
        field.print(std::cerr);
        continue;
      }
      size_t id = field_ids[cmd];
      field.put(id, x, y);
    } else if (cmd == "pop") {
      field.pop();
    } else if (cmd == "r") {
      field.rotate();
    } else if (cmd == "l") {
      for (const auto& detail : details) {
        detail.print(std::cerr);
      }
    } else if (cmd.front() == '4') {
      --y;
    } else if (cmd.front() == '6') {
      ++y;
    } else if (cmd.front() == '8') {
      --x;
    } else if (cmd.front() == '5') {
      ++x;
    } else if (cmd == "m") {
      field.match();
    } else if (cmd == "mask") {
      int x = 0;
      int y = 0;
      std::cerr << "\033[2J";
      field.showmask(std::cerr);
      if (std::cin >> x >> y) {
        std::cerr << "set mask: " << x << " " << y << "\n";
        field.putmask(x, y);
        field.showmask(std::cerr);
        continue;
      }
    } else if (cmd == "clrmask") {
      int x = 0;
      int y = 0;
      std::cerr << "\033[2J";
      field.showmask(std::cerr);
      if (std::cin >> x >> y) {
        std::cerr << "clear mask: " << x << " " << y << "\n";
        field.clrmask(x, y);
        field.showmask(std::cerr);
        continue;
      }
    } else if (cmd == "showmask") {
      field.showmask(std::cerr);
      continue;
    }
    field.print(std::cerr);
  }
  // field.put(details[1], 3, 5);
  field.print(std::cerr);
  // field.checksen();
  // field.pop(details[0], 0, 0);
  field.Grid::print(std::cerr);
  std::cout << "Hello, from fitit!\n";
}
