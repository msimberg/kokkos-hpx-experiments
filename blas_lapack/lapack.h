#ifndef LAPACK_H
#define LAPACK_H

extern "C" {
void spotrf_(const char* uplo, const int* n, float* a, const int* lda, int* info);
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
void cpotrf_(const char* uplo, const int* n, std::complex<float>* a, const int* lda, int* info);
void zpotrf_(const char* uplo, const int* n, std::complex<double>* a, const int* lda, int* info);

void slaset_(const char* uplo, const int* m, const int* n, const float* alpha, const float* beta,
             float* a, const int* lda);
void dlaset_(const char* uplo, const int* m, const int* n, const double* alpha, const double* beta,
             double* a, const int* lda);
void claset_(const char* uplo, const int* m, const int* n, const std::complex<float>* alpha,
             const std::complex<float>* beta, std::complex<float>* a, const int* lda);
void zlaset_(const char* uplo, const int* m, const int* n, const std::complex<double>* alpha,
             const std::complex<double>* beta, std::complex<double>* a, const int* lda);

void slacpy_(const char* uplo, const int* m, const int* n, const float* a, const int* lda, float* b,
             const int* ldb);
void dlacpy_(const char* uplo, const int* m, const int* n, const double* a, const int* lda,
             double* b, const int* ldb);
void clacpy_(const char* uplo, const int* m, const int* n, const std::complex<float>* a,
             const int* lda, std::complex<float>* b, const int* ldb);
void zlacpy_(const char* uplo, const int* m, const int* n, const std::complex<double>* a,
             const int* lda, std::complex<double>* b, const int* ldb);

float slange_(const char* norm, const int* m, const int* n, const float* a, const int* lda,
              float* work);
double dlange_(const char* norm, const int* m, const int* n, const double* a, const int* lda,
               double* work);
float clange_(const char* norm, const int* m, const int* n, const std::complex<float>* a,
              const int* lda, float* work);
double zlange_(const char* norm, const int* m, const int* n, const std::complex<double>* a,
               const int* lda, double* work);

float slansy_(const char* norn, const char* uplo, const int* n, const float* a, const int* lda,
              float* work);
double dlansy_(const char* norn, const char* uplo, const int* n, const double* a, const int* lda,
               double* work);
float clansy_(const char* norn, const char* uplo, const int* n, const std::complex<float>* a,
              const int* lda, float* work);
double zlansy_(const char* norn, const char* uplo, const int* n, const std::complex<double>* a,
               const int* lda, double* work);

float clanhe_(const char* norm, const char* uplo, const int* n, const std::complex<float>* a,
              const int* lda, float* work);
double zlanhe_(const char* norm, const char* uplo, const int* n, const std::complex<double>* a,
               const int* lda, double* work);
}

inline void potrf(const char* uplo, int n, float* a, int lda, int* info) {
  spotrf_(uplo, &n, a, &lda, info);
}
inline void potrf(const char* uplo, int n, double* a, int lda, int* info) {
  dpotrf_(uplo, &n, a, &lda, info);
}
inline void potrf(const char* uplo, int n, std::complex<float>* a, int lda, int* info) {
  cpotrf_(uplo, &n, a, &lda, info);
}
inline void potrf(const char* uplo, int n, std::complex<double>* a, int lda, int* info) {
  zpotrf_(uplo, &n, a, &lda, info);
}

inline void laset(const char* uplo, int m, int n, float alpha, float beta, float* a, int lda) {
  slaset_(uplo, &m, &n, &alpha, &beta, a, &lda);
}
inline void laset(const char* uplo, int m, int n, double alpha, double beta, double* a, int lda) {
  dlaset_(uplo, &m, &n, &alpha, &beta, a, &lda);
}
inline void laset(const char* uplo, int m, int n, std::complex<float> alpha,
                  std::complex<float> beta, std::complex<float>* a, int lda) {
  claset_(uplo, &m, &n, &alpha, &beta, a, &lda);
}
inline void laset(const char* uplo, int m, int n, std::complex<double> alpha,
                  std::complex<double> beta, std::complex<double>* a, int lda) {
  zlaset_(uplo, &m, &n, &alpha, &beta, a, &lda);
}

inline void lacpy(const char* uplo, int m, int n, const float* a, int lda, float* b, int ldb) {
  slacpy_(uplo, &m, &n, a, &lda, b, &ldb);
}
inline void lacpy(const char* uplo, int m, int n, const double* a, int lda, double* b, int ldb) {
  dlacpy_(uplo, &m, &n, a, &lda, b, &ldb);
}
inline void lacpy(const char* uplo, int m, int n, const std::complex<float>* a, int lda,
                  std::complex<float>* b, int ldb) {
  clacpy_(uplo, &m, &n, a, &lda, b, &ldb);
}
inline void lacpy(const char* uplo, int m, int n, const std::complex<double>* a, int lda,
                  std::complex<double>* b, int ldb) {
  zlacpy_(uplo, &m, &n, a, &lda, b, &ldb);
}

inline float lange(const char* norm, int m, int n, const float* a, int lda, float* work) {
  return slange_(norm, &m, &n, a, &lda, work);
}
inline double lange(const char* norm, int m, int n, const double* a, int lda, double* work) {
  return dlange_(norm, &m, &n, a, &lda, work);
}
inline float lange(const char* norm, int m, int n, const std::complex<float>* a, int lda,
                   float* work) {
  return clange_(norm, &m, &n, a, &lda, work);
}
inline double lange(const char* norm, int m, int n, const std::complex<double>* a, int lda,
                    double* work) {
  return zlange_(norm, &m, &n, a, &lda, work);
}

inline float lansy(const char* norn, const char* uplo, int n, const float* a, int lda, float* work) {
  return slansy_(norn, uplo, &n, a, &lda, work);
}
inline double lansy(const char* norn, const char* uplo, int n, const double* a, int lda,
                    double* work) {
  return dlansy_(norn, uplo, &n, a, &lda, work);
}
inline float lansy(const char* norn, const char* uplo, int n, const std::complex<float>* a, int lda,
                   float* work) {
  return clansy_(norn, uplo, &n, a, &lda, work);
}
inline double lansy(const char* norn, const char* uplo, int n, const std::complex<double>* a,
                    int lda, double* work) {
  return zlansy_(norn, uplo, &n, a, &lda, work);
}

inline float lanhe(const char* norn, const char* uplo, int m, int n, const float* a, int lda,
                   float* work) {
  return slansy_(norn, uplo, &n, a, &lda, work);
}
inline double lanhe(const char* norn, const char* uplo, int n, const double* a, int lda,
                    double* work) {
  return dlansy_(norn, uplo, &n, a, &lda, work);
}
inline float lanhe(const char* norm, const char* uplo, int n, const std::complex<float>* a, int lda,
                   float* work) {
  return clanhe_(norm, uplo, &n, a, &lda, work);
}
inline double lanhe(const char* norm, const char* uplo, int n, const std::complex<double>* a,
                    int lda, double* work) {
  return zlanhe_(norm, uplo, &n, a, &lda, work);
}

#endif
