#include "NeuralNetwork.h"

// ========== 学習したニューラルネットワークの構造を記述 ==========

// 深さ = 中間層の数+入力層+出力層
#define depth 1+2
// ノードが一番多い層のノード数
#define MaxNode 3
//各層のノード数 {入力層, 隠れ層, ... ,出力層}
const uint8_t nodeNum[depth] = {2,3,1};

// ========== 重み、バイアスの学習結果をコピペ ==========
// ===== ここからコピペ =====
const float w[depth-1][MaxNode][MaxNode] = {
{{ 0.72673529,  1.3485838 , -0.79594878},
 {-0.0321573 , -1.45074034,  1.25377813}},
{{-0.57552841},
 { 0.63916822},
 { 1.72330261}}
};
const float bias[depth-1][MaxNode] = {
{-0.89842617,  0.60438132, -0.48385345},
{-0.31654861}
};
// ===== ここまでコピペ =====

NeuralNetwork::NeuralNetwork(){
}

float NeuralNetwork::relu(float inputSum){
  if(inputSum<0) return 0;
  else return inputSum;
}

float NeuralNetwork::getWeight(uint8_t layer, uint8_t node, uint8_t output){
  return w[layer][node][output];
}

float NeuralNetwork::getBias(uint8_t layer, uint8_t node){
  return bias[layer][node];
}


// ========== ニューラルネットワークで計算(推論) ==========

void NeuralNetwork::predict(float *inputs, float *outputs){
  //最大ノード数の配列を作成し、入力層の値をセット
  //float prevLayerNode[MaxNode] = {1,2,3,4};
  float prevLayerNode[MaxNode];
  for(int i=0; i<nodeNum[0]; i++){
    prevLayerNode[i] = inputs[i];
  }

  //層ごとに計算
  for(int layer=0; layer<=depth-2; layer++){
    double nextLayerNode[MaxNode] = {0};
    //各層のノードごとに計算
    for(int node=0; node<=nodeNum[layer]-1; node++){
      //次層のノード数 = そのノードから出力されるデータ数
      for(int output=0; output<=nodeNum[layer+1]-1; output++){
        //flashに格納されている重みを取得
        float w = getWeight(layer,node,output);
        //層に格納されている値と重みの積を次の層に格納
        nextLayerNode[output] += prevLayerNode[node] * w;
      }
    }
    for(int node=0; node<=nodeNum[layer+1]-1; node++){
      //バイアス
      nextLayerNode[node] += getBias(layer,node);
      //活性化関数
      nextLayerNode[node] = relu(nextLayerNode[node]);
      //コピー
      prevLayerNode[node] = nextLayerNode[node];
    }
  }

  //出力層の結果を、出力用の配列に格納
  for(int i=0; i<nodeNum[depth-1]; i++){
    outputs[i] = prevLayerNode[i];
  }
}
