#include <array>
#include <bitset>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <vector>

struct Detail {
  Detail(std::initializer_list<std::string> rows) {
    for (const auto& row : rows) {
      W = std::max(W, row.size() >> 1);
      size_t chunk = 0;
      for (auto it = row.rbegin(); it < row.rend(); it += 2) {
        chunk = (chunk << 1) | (*it != ' ');
      }
      field.push_back(chunk);
    }
    print(std::cerr);
  }

  void print(std::ostream& os) const {
    bool even = true;
    for (size_t row : field) {
      for (size_t cell = 0; cell < W; ++cell) {
        if (row & (1ull << cell)) {
          if (even ^ (cell & 1)) {
            os << "()";
          } else {
            os << (even ? "><" : "><");
          }
        } else {
          os << "  ";
        }
      }
      os << "\n";
      even ^= true;
    }
    os << "\n";
  }

 private:
  std::vector<size_t> field;
  size_t W = 0;
  friend class Field;
};

class Field {
 public:
  static constexpr size_t W = 35;
  static constexpr size_t H = 35;

  Field() { field.fill(0); }

  void print(std::ostream& os) const {
    bool even = true;
    for (size_t row : field) {
      for (size_t cell = 0; cell < W; ++cell) {
        if (row & (1ull << cell)) {
          if (even ^ (cell & 1)) {
            os << "()";
          } else {
            os << (even ? "><" : "><");
          }
        } else {
          os << "  ";
        }
      }
      os << "\n";
      even ^= true;
    }
  }

  bool put(const Detail& detail, int x, int y) {
    if (x + detail.field.size() > field.size()) return false;
    const int base_x = x;
    for (const auto& row : detail.field) {
      if ((row << y) & field[x++]) return false;
    }
    x = base_x;
    for (const auto& row : detail.field) {
      field[x++] |= (row << y);
    }
    return true;
  }


  void pop(const Detail& detail, int x, int y) {
    for (const auto& row : detail.field) {
      field[x++] &= ~(row << y);
    }
  }

 private:
  std::array<size_t, H> field;
};

// {
//   "()==()==()",
//   "||  ||  ||",
//   "()  ()  ()",
// }
std::vector<Detail> details = {
    Detail({
        "    ()==()              ",  //
        "    ||                  ",  //
        "    ()==()==()==()==()==",  //
        "||  ||  ||          ||  ",  //
        "()==()  ()==()==()  ()  ",  //
    }),
    Detail({
        "()==()==()  ()",  //
        "        ||  ||",  //
        "        ()==()",  //
    }),
    Detail({

        "    ()  ()            ",  //
        "    ||  ||            ",  //
        "()==()==()==()==()==()",  //
        "    ||  ||            ",  //
        "        ()==()==()    ",  //
        "        ||            ",  //
        "        ()==()        ",  //
        "            ||        ",  //
        "          ==()==()==  ",  //
    }),
    Detail({

        "()==()==()    ",  //
        "        ||    ",  //
        "      ==()==()",  //
    }),
    Detail({

        "()==()==()==()",  //
        "        ||    ",  //
        "    ()==()==()",  //
        "    ||        ",  //
        "    ()==()==()",  //
        "        ||    ",  //
        "()==()==()    ",  //
        "    ||        ",  //
        "()==()==()    ",  //
        "    ||        ",  //
        "    ()==()    ",  //
    }),
    Detail({

        "        ()==()    ",  //
        "            ||    ",  //
        "        ()==()==()",  //
        "            ||    ",  //
        "()==()==()==()==()",  //
    }),
    Detail({

        "()==()==()==()==()==()",  //
        "        ||  ||        ",  //
        "        ()  ()==()    ",  //
        "                ||    ",  //
        "                ()==  ",  //
    }),

};

int main(int, char**) {
  Field field;
  field.put(details[0], 0, 0);
  field.put(details[1], 3, 5);

  auto checksen = [&field](auto self, int d) -> void {
    for (int x = 0; x < Field::H; ++x) {
      for (int y = 0; y < Field::W; ++y) {
        if (!field.put(details[d], x, y)) continue;
        if (d + 1 != details.size()) {
          self(self, d + 1);
        } else {
          field.print(std::cerr);
        }
        field.pop(details[d], x, y);
      }
    }
  };
  for (int d = 0; d < details.size(); ++d) {
    checksen(checksen, d);
  }

  // field.pop(details[0], 0, 0);
  field.print(std::cerr);
  std::cout << "Hello, from fitit!\n";
}
