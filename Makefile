CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -funroll-loops -Wall -Wextra
TARGET = KrishLLM
SRC = main.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
