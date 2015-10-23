#include "SRNN.hpp"

/* コンストラクタ - 最小の初期化パラメタ
 * 適宜追加する可能性あり
 */
SRNN::SRNN(int    dim, 
           int    num_mid,
           int    len_seq,
           float* input_sample,
           float* input_sample_maxmin)
{
  
  this->dim_signal = dim;
  this->num_mid_neuron = num_mid; // Advice : number of hidden layter shuld be as large as possible.
  this->len_seqence = len_seq;
  
  // sample/sample_maxmin allocation
  this->sample = new float[len_seqence * dim_signal];
  this->sample_maxmin = new float[dim_signal * 2];

  memcpy(this->sample, input_sample, sizeof(float) * len_seqence * dim_signal);
  memcpy(this->sample_maxmin, input_sample_maxmin, sizeof(float) * dim_signal * 2);
        
  this->predict_signal = new float[dim_signal];
  
  // coffecience matrix allocation
  // final +1 for bias 
  this->Win_mid  = new float[num_mid_neuron * (dim_signal + num_mid_neuron + 1)]; 
  this->Wmid_out = new float[dim_signal * (num_mid_neuron + 1)];
  

  // input/hidden layer signal allocation
  expand_in_signal = new float[dim_signal + num_mid_neuron + 1];
  expand_mid_signal = new float[num_mid_neuron + 1];
  
  // Parameter settings (Tuning by taiyo)
  this->squareError    = FLT_MAX; // (large value)
  this->maxIteration   = 5000;
  this->goalError      = float(0.001);
  this->epsilon        = float(0.00001);
  this->learnRate      = float(0.1);   
  this->alpha          = float(0.8 * learnRate);
  this->alpha_context  = float(0.8);
  this->width_initW    = float(1.0/num_mid_neuron);

  // random seed decide by time
  srand((unsigned int)time(NULL));
  
}

SRNN::~SRNN(void)
{
    delete [] sample; delete [] sample_maxmin;
    delete [] predict_signal;
    delete [] Win_mid; delete [] Wmid_out;
    delete [] expand_in_signal;
    delete [] expand_mid_signal;
}

/* utilにいどうするべき */
void SRNN::sigmoid_vec(float* net,
                       float* out,
                       int    dim)
{
  for (int n=0;n<dim;n++)
    out[n] = sigmoid_func(net[n]);
}

/* Predict : predicting next sequence of input */
void SRNN::predict(float* input)
{
  float *norm_input = new float[this->dim_signal];

  // normalize signal
  for (int n=0; n < dim_signal; n++) {
    norm_input[n] = 
      normalize_signal(input[n],
          MATRIX_AT(this->sample_maxmin,2,n,0),
          MATRIX_AT(this->sample_maxmin,2,n,1));
  }

  // output signal
  float* out_signal = new float[dim_signal];
  // value of network in input->hidden layer 
  float* in_mid_net = new float[num_mid_neuron];
  // value of network in hidden->output layer 
  float* mid_out_net = new float[dim_signal];

  /* Calcurate output signal */
  // Get input signal 
  memcpy(expand_in_signal, norm_input, sizeof(float) * dim_signal);
  // Signal of input layer : 中間層との線形和をシグモイド関数に通す.
  for (int d = 0; d < num_mid_neuron; d++) {
    expand_in_signal[dim_signal + d] = sigmoid_func(alpha_context * expand_in_signal[dim_signal + d] + expand_mid_signal[d]);
  }
  // Bias fixed at 1.
  expand_in_signal[dim_signal + num_mid_neuron] = 1;

  // 入力->中間層の出力信号和計算
  multiply_mat_vec(Win_mid, expand_in_signal, in_mid_net, num_mid_neuron, dim_signal + num_mid_neuron + 1);
  // 中間層の出力信号計算
  sigmoid_vec(in_mid_net, expand_mid_signal, num_mid_neuron);
  expand_mid_signal[num_mid_neuron] = 1;

  // 中間->出力層の出力信号和計算
  multiply_mat_vec(Wmid_out, expand_mid_signal, mid_out_net, dim_signal, num_mid_neuron + 1);
  // 出力層の出力信号計算
  sigmoid_vec(mid_out_net, out_signal, dim_signal);

  // expand output signal to origin width.
  for (int n=0;n < dim_signal;n++) {
    predict_signal[n] = expand_signal(out_signal[n],sample_maxmin[n * 2],sample_maxmin[n * 2 + 1]);
  }
  
  delete [] norm_input; delete [] out_signal;
  delete [] in_mid_net; delete [] mid_out_net;

}

/* 逆誤差伝搬法による学習 局所解？なんのこったよ（すっとぼけ）*/
float SRNN::learning(void)
{
  int iteration = 0; // 学習繰り返し回数
  int seq = 0;       // 現在学習中の系列番号[0,...,len_seqence-1]
  int end_flag = 0;  // 学習終了フラグ.このフラグが成立したら今回のsequenceを最後まで回して終了する.
  // 係数行列のサイズ
  int row_in_mid = num_mid_neuron;
  int col_in_mid = dim_signal + num_mid_neuron + 1;
  int row_mid_out = dim_signal;
  int col_mid_out = num_mid_neuron + 1;

  // 行列のアロケート
  // 係数行列の更新量
  float* dWin_mid  = new float[row_in_mid * col_in_mid];
  float* dWmid_out = new float[row_mid_out * col_mid_out];
  // 前回の更新量:慣性項に用いる.
  float* prevdWin_mid  = new float[row_in_mid * col_in_mid];
  float* prevdWmid_out = new float[row_mid_out * col_mid_out];
  float* norm_sample   = new float[len_seqence * dim_signal]; // 正規化したサンプル信号; 実際の学習は正規化した信号を用います.

  // 係数行列の初期化
  for (int i=0; i < row_in_mid; i++)
    for (int j=0; j < col_in_mid; j++)
      MATRIX_AT(Win_mid,col_in_mid,i,j) = uniform_rand(width_initW);

  for (int i=0; i < row_mid_out; i++)
    for (int j=0; j < col_mid_out; j++)
      MATRIX_AT(Wmid_out,col_mid_out,i,j) = uniform_rand(width_initW);

  // 信号の正規化:経験上,非常に大切な処理
  for (int seq=0; seq < len_seqence; seq++) {
    for (int n=0; n < dim_signal; n++) {
      MATRIX_AT(norm_sample,dim_signal,seq,n) = 
            normalize_signal(MATRIX_AT(this->sample,dim_signal,seq,n),
                             MATRIX_AT(this->sample_maxmin,2,n,0),
                             MATRIX_AT(this->sample_maxmin,2,n,1));
      // printf("%f ", MATRIX_AT(norm_sample,dim_signal,seq,n));
    }
    // printf("\r\n");
  }

  // 出力層の信号
  float* out_signal = new float[dim_signal];

  // 入力層->中間層の信号和
  float* in_mid_net = new float[num_mid_neuron];
  // 中間層->出力層の信号和.
  float* mid_out_net = new float[dim_signal];

  // 誤差信号
  float* sigma = new float[dim_signal];

  // 前回の二乗誤差値:収束判定に用いる.
  float prevError;

  /* 学習ループ */
  while (1) {

    // 終了条件を満たすか確認
    if (!end_flag) {
      end_flag = !(iteration < this->maxIteration 
                   && (iteration <= this->len_seqence 
                       || this->squareError > this->goalError)
                  );
    }

    // printf("ite:%d err:%f \r\n", iteration, squareError);

    // 系列の末尾に到達していたら,最初からリセットする.
    if (seq == len_seqence && !end_flag) {
      seq = 0;
    }

    // 前回の更新量/二乗誤差を保存
    if (iteration >= 1) {
      memcpy(prevdWin_mid, dWin_mid, sizeof(float) * row_in_mid * col_in_mid);
      memcpy(prevdWmid_out, dWmid_out, sizeof(float) * row_mid_out * col_mid_out);
      prevError = squareError;
    } else {
      // 初回は0埋め
      memset(prevdWin_mid, float(0), sizeof(float) * row_in_mid * col_in_mid);
      memset(prevdWmid_out, float(0), sizeof(float) * row_mid_out * col_mid_out);
    }
    
    /* 学習ステップその1:ニューラルネットの出力信号を求める */

    // 入力値を取得
    memcpy(expand_in_signal, &(norm_sample[seq * dim_signal]), sizeof(float) * dim_signal);
    // SRNN特有:入力層に中間層のコピーが追加され,中間層に入力される.
    if (iteration == 0) {
      // 初回は0埋めする
      memset(&(expand_in_signal[dim_signal]), float(0), sizeof(float) * num_mid_neuron);
    } else {
      // コンテキスト層 = 前回のコンテキスト層の出力
      // 前回の中間層信号との線形和をシグモイド関数に通す.
      for (int d = 0; d < num_mid_neuron; d++) {
        expand_in_signal[dim_signal + d] = sigmoid_func(alpha_context * expand_in_signal[dim_signal + d] + expand_mid_signal[d]);
      }
    }
    // バイアス項は常に1に固定.
    expand_in_signal[dim_signal + num_mid_neuron] = 1;

    // 入力->中間層の出力信号和計算
    multiply_mat_vec(Win_mid,
                     expand_in_signal,
                     in_mid_net,
                     num_mid_neuron,
                     dim_signal + num_mid_neuron + 1);
    // 中間層の出力信号計算
    sigmoid_vec(in_mid_net,
                expand_mid_signal,
                num_mid_neuron);
    expand_mid_signal[num_mid_neuron] = 1;
    // 中間->出力層の出力信号和計算
    multiply_mat_vec(Wmid_out,
                     expand_mid_signal,
                     mid_out_net,
                     dim_signal,
                     num_mid_neuron + 1);
    // 出力層の出力信号計算
    sigmoid_vec(mid_out_net,
                out_signal,
                dim_signal);

    
    for (int i = 0; i < dim_signal; i++) {
      predict_signal[i] = expand_signal(out_signal[i],
                                        MATRIX_AT(sample_maxmin,2,i,0),
                                        MATRIX_AT(sample_maxmin,2,i,1));
    }
    printf("predict : %f %f %f \r\n", predict_signal[0], predict_signal[1], predict_signal[2]);
    
    // print_mat(Wmid_out, row_mid_out, col_mid_out);
    
    // この時点での二乗誤差計算
    squareError = 0;
    // 次の系列との誤差を見ている!! ここが注目ポイント
    // ==> つまり,次系列を予測させようとしている.
    for (int n = 0;n < dim_signal;n++) {
      if (seq < len_seqence - 1) {
        squareError += powf((out_signal[n] - MATRIX_AT(norm_sample,dim_signal,(seq + 1),n)),2);
      } else {
        squareError += powf((out_signal[n] - MATRIX_AT(norm_sample,dim_signal,0,n)),2);
      }
    } 
    squareError /= dim_signal;

    /* 学習の終了 */
    // 終了フラグが立ち,かつ系列の最後に達していたら学習終了
    if (end_flag && (seq == (len_seqence-1))) {
      // 予測結果をセット.
      for (int i = 0; i < dim_signal; i++) {
        predict_signal[i] = expand_signal(out_signal[i],
                                          MATRIX_AT(sample_maxmin,2,i,0),
                                          MATRIX_AT(sample_maxmin,2,i,1));
        //printf("%f ", predict_signal[i]);
      }
      break;
    }

    // 収束したと判定したら終了フラグを立てる.
    if (fabsf(squareError - prevError) < epsilon) {
      end_flag = 1;
    }

    /* 学習ステップその2:逆誤差伝搬 */
    // 誤差信号の計算
    for (int n = 0; n < dim_signal; n++) {
      if (seq < len_seqence - 1) {
        sigma[n] = (out_signal[n] - MATRIX_AT(norm_sample,dim_signal,seq+1,n)) * out_signal[n] * (1 - out_signal[n]);
      } else {
        /* 末尾と先頭の誤差を取る (大抵,大きくなる) */
        sigma[n] = (out_signal[n] - MATRIX_AT(norm_sample, dim_signal,0,n)) * out_signal[n] * (1 - out_signal[n]);
      }
    }
    // printf("Sigma : %f %f %f \r\n", sigma[0], sigma[1], sigma[2]);

    // 出力->中間層の係数の変更量計算
    for (int n = 0; n < dim_signal; n++) {
      for (int j = 0; j < num_mid_neuron + 1; j++) {
        MATRIX_AT(dWmid_out,num_mid_neuron,n,j) = sigma[n] * expand_mid_signal[j];
      }
    }

    // 中間->入力層の係数の変更量計算
    register float sum_sigma;
    for (int i = 0; i < num_mid_neuron; i++) {
      // 誤差信号を逆向きに伝播させる.
      sum_sigma = 0;
      for (int k = 0; k < dim_signal; k++) {
        sum_sigma += sigma[k] * MATRIX_AT(Wmid_out,num_mid_neuron + 1,k,i);
      }
      // 中間->入力層の係数の変更量計算
      for (int j = 0; j < col_in_mid; j++) {
        MATRIX_AT(dWin_mid,num_mid_neuron,j,i)
                          = sum_sigma * expand_mid_signal[i] *
                            (1 - expand_mid_signal[i]) *
                            expand_in_signal[j];
      }
    }

    // 係数更新
    for (int i = 0; i < row_in_mid; i++) {
      for (int j = 0; j < col_in_mid; j++) {
        //printf("[%f -> ", MATRIX_AT(Win_mid,col_in_mid,i,j));
        MATRIX_AT(Win_mid,col_in_mid,i,j) = 
              MATRIX_AT(Win_mid,col_in_mid,i,j) - 
              this->learnRate * MATRIX_AT(dWin_mid,col_in_mid,i,j) -
              this->alpha * MATRIX_AT(prevdWin_mid,col_in_mid,i,j);
        // printf("%f] ", MATRIX_AT(Win_mid,col_in_mid,i,j));
        // printf("dW : %f , prevdW : %f ", MATRIX_AT(dWin_mid,col_in_mid,i,j), MATRIX_AT(prevdWin_mid,col_in_mid,i,j));
      }
      //printf("\r\n");
    }
    for (int i = 0; i < row_mid_out; i++) {
      for (int j = 0; j < col_mid_out; j++) {
        MATRIX_AT(Wmid_out,col_mid_out,i,j)= 
              MATRIX_AT(Wmid_out,col_mid_out,i,j) - 
              this->learnRate * MATRIX_AT(dWmid_out,col_mid_out,i,j) - 
              this->alpha * MATRIX_AT(prevdWmid_out,col_mid_out,i,j);
      }
    }

    // ループ回数/系列のインクリメント
    iteration += 1;
    seq += 1;

  }
  
  delete [] dWin_mid; delete [] dWmid_out;
  delete [] prevdWin_mid; delete [] prevdWmid_out;
  delete [] norm_sample; delete [] out_signal;
  delete [] in_mid_net; delete [] mid_out_net;

  return squareError;
}
