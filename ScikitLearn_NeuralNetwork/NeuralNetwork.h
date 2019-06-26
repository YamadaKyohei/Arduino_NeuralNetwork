#include "Arduino.h"

class NeuralNetwork {
public:
  NeuralNetwork(void);
  void predict(float*, float*);
private:
  float relu(float inputSum);
  float getWeight(uint8_t layer, uint8_t node, uint8_t output);
  float getBias(uint8_t layer, uint8_t node);
};
