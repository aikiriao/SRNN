#ifndef SRNN_H_INCLUDED
#define SRNN_H_INCLUDED

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "util.hpp"

class SRNN
{
  private:
    int    dim_signal;            // 入力次元=出力次元数
    int    num_mid_neuron;        // 中間層のニューロン数
    float  width_initW;           // 係数行列初期値の乱数幅:[-width_initW,+width_initW]
    float  goalError;             // 二乗誤差の目標値
    float  epsilon;               // 収束判定用の小さい値
    int    maxIteration;          // 最大学習繰り返し回数
    float  learnRate;             // 学習係数(must be in [0,1])
    float  alpha;                 // 慣性項の係数(0.8 * learnRateぐらいに設定する予定)
    float  alpha_context;         // コンテキスト層の重み付け[0,1]
    int    len_seqence;           // サンプルの系列長
    float* Win_mid;               // 入力<->中間層の係数行列
    float* Wmid_out;              // 中間<->出力層の係数行列
    float* expand_in_signal;      // コンテキスト層も含めた入力層の出力信号.SRNN特有
    float* expand_mid_signal;     // 中間層の出力信号 
  public:
    float  squareError;           // 二乗誤差(経験誤差)
    float* sample;                // 系列長len_seqenceに渡る次元dim_signalのサンプル
    float* sample_maxmin;         // サンプルの取りうる最大/最小値信号を並べたベクトル
    float* predict_signal;        // 予測出力

  private:  
    // サイズn*1のベクトルの要素をそれぞれシグモイド関数に通して,
    // 結果をoutにセットする.(ニューロンのユニット動作を一括で)
    void sigmoid_vec(float*,   // n * 1 入力ベクトル
                     float*,   // n * 1 出力ベクトル
                     int);     // n
                     
  public:
    // 最小限の初期化パラメタによるコンストラクタ.配列(ベクトル)のアロケートを行う.
    // 適宜追加する予定
    SRNN(int,          // 信号の次元dim_signal
         int,          // 中間層の数num_mid_neuron
         int,          // 系列長len_seqence
         float*,       // サンプル
         float*);      // サンプルの最大値/最小値ベクトル
    
    ~SRNN(void);

    // 逆誤差伝搬法による学習を行い,経験誤差が目標値goalErrorに達するか,
    // 最大繰り返し回数maxIterationに到達したら,その時の二乗誤差を出力する.
    float learning(void); 

    // 予測結果predict_signalにセット
    void predict(float *input); 
};

#endif /* SRNN_H_INCLUDED */
