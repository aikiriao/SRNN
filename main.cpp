#include "mbed.h"

#include "SRNN.hpp"
#include "./debug/debug.hpp"

LocalFileSystem local("local");  // マウントポイントを定義（ディレクトリパスになる）

// SRNNのテストルーチン
int main(void) {
  FILE* fp;
  const char* fname = "/local/testdata.csv";
  // char s[100];
  // int ret, n1, n2;
  int ret;
  float f1, f2, f3;
  
  set_new_handler(no_memory);

  float* test_sample = new float[250 * 3];
  float* test_sample_maxmin = new float[3 * 2];
  test_sample_maxmin[0] = 20;
  test_sample_maxmin[1] = 0;
  test_sample_maxmin[2] = 1025;
  test_sample_maxmin[3] = 980;
  test_sample_maxmin[4] = 20;
  test_sample_maxmin[5] = 0;

  fp = fopen( fname, "r" );
  if( fp == NULL ){
    printf( "File[%s] cannot open. \r\n", fname );
    return 1;
  }

  int cnt = 0;
  while( ( ret = fscanf( fp, "%f,%f,%f", &f1, &f2, &f3) ) != EOF ){
    // printf( "%f %f %f \n", f1, f2, f3 );
    test_sample[cnt * 3] = f1;
    test_sample[cnt * 3 + 1] = f2;
    test_sample[cnt * 3 + 2] = f3;
    // printf("sample : %f %f %f \r\n", MATRIX_AT(test_sample,3,cnt,0), MATRIX_AT(test_sample,3,cnt,1), MATRIX_AT(test_sample,3,cnt,2));
    cnt++;
  }

  /* アドバイス:RNNにおいては,ダイナミクス(中間層のニューロン数)は多いほど良い */
  SRNN srnn(3, 30, 240, test_sample, test_sample_maxmin);

  srnn.learning();

  delete [] test_sample; delete [] test_sample_maxmin;
  fclose( fp );
  free( fp );   // required for mbed.
  return 0;
  
}

