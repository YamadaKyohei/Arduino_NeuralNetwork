#include "NeuralNetwork.h"

// ライブラリ内で定義したニューラルネットワークを作成
NeuralNetwork myNeuralNet;

void setup() {
  Serial.begin(9600);

  // 計算（推論）してみる
  float input[2] = {0, 0}; // 入力層
  float output[1]; // 出力層
  myNeuralNet.predict(input, output);
  Serial.println(output[0],5);

  input[0] = 0; // 入力
  input[1] = 1;
  myNeuralNet.predict(input, output);
  Serial.println(output[0],5);

  input[0] = 1; // 入力
  input[1] = 0;
  myNeuralNet.predict(input, output);
  Serial.println(output[0],5);

  input[0] = 1; // 入力
  input[1] = 1;
  myNeuralNet.predict(input, output);
  Serial.println(output[0],5);
}

void loop() {
}
