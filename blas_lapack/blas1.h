#ifndef BLAS1_H
#define BLAS1_H

extern "C" {
void sswap_(const int* n, float* x, const int* incx, float* y, const int* incy);
void dswap_(const int* n, double* x, const int* incx, double* y, const int* incy);
void cswap_(const int* n, std::complex<float>* x, const int* incx, std::complex<float>* y,
            const int* incy);
void zswap_(const int* n, std::complex<double>* x, const int* incx, std::complex<double>* y,
            const int* incy);

void sscal_(const int* n, const float* alpha, float* y, const int* incy);
void dscal_(const int* n, const double* alpha, double* y, const int* incy);
void cscal_(const int* n, const std::complex<float>* alpha, std::complex<float>* y, const int* incy);
void zscal_(const int* n, const std::complex<double>* alpha, std::complex<double>* y,
            const int* incy);

void scopy_(const int* n, const float* x, const int* incx, float* y, const int* incy);
void dcopy_(const int* n, const double* x, const int* incx, double* y, const int* incy);
void ccopy_(const int* n, const std::complex<float>* x, const int* incx, std::complex<float>* y,
            const int* incy);
void zcopy_(const int* n, const std::complex<double>* x, const int* incx, std::complex<double>* y,
            const int* incy);

void saxpy_(const int* n, const float* alpha, const float* x, const int* incx, float* y,
            const int* incy);
void daxpy_(const int* n, const double* alpha, const double* x, const int* incx, double* y,
            const int* incy);
void caxpy_(const int* n, const std::complex<float>* alpha, const std::complex<float>* x,
            const int* incx, std::complex<float>* y, const int* incy);
void zaxpy_(const int* n, const std::complex<double>* alpha, const std::complex<double>* x,
            const int* incx, std::complex<double>* y, const int* incy);

float sdot_(const int* n, const float* x, const int* incx, const float* y, const int* incy);
double ddot_(const int* n, const double* x, const int* incx, const double* y, const int* incy);
}

inline void swap(int n, float* x, int incx, float* y, int incy) {
  sswap_(&n, x, &incx, y, &incy);
}
inline void swap(int n, double* x, int incx, double* y, int incy) {
  dswap_(&n, x, &incx, y, &incy);
}
inline void swap(int n, std::complex<float>* x, int incx, std::complex<float>* y, int incy) {
  cswap_(&n, x, &incx, y, &incy);
}
inline void swap(int n, std::complex<double>* x, int incx, std::complex<double>* y, int incy) {
  zswap_(&n, x, &incx, y, &incy);
}

inline void scal(int n, float alpha, float* y, int incy) {
  sscal_(&n, &alpha, y, &incy);
}
inline void scal(int n, double alpha, double* y, int incy) {
  dscal_(&n, &alpha, y, &incy);
}
inline void scal(int n, std::complex<float> alpha, std::complex<float>* y, int incy) {
  cscal_(&n, &alpha, y, &incy);
}
inline void scal(int n, std::complex<double> alpha, std::complex<double>* y, int incy) {
  zscal_(&n, &alpha, y, &incy);
}

inline void copy(int n, const float* x, int incx, float* y, int incy) {
  scopy_(&n, x, &incx, y, &incy);
}
inline void copy(int n, const double* x, int incx, double* y, int incy) {
  dcopy_(&n, x, &incx, y, &incy);
}
inline void copy(int n, const std::complex<float>* x, int incx, std::complex<float>* y, int incy) {
  ccopy_(&n, x, &incx, y, &incy);
}
inline void copy(int n, const std::complex<double>* x, int incx, std::complex<double>* y, int incy) {
  zcopy_(&n, x, &incx, y, &incy);
}

inline void axpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
  saxpy_(&n, &alpha, x, &incx, y, &incy);
}
inline void axpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
  daxpy_(&n, &alpha, x, &incx, y, &incy);
}
inline void axpy(int n, std::complex<float> alpha, const std::complex<float>* x, int incx,
                 std::complex<float>* y, int incy) {
  caxpy_(&n, &alpha, x, &incx, y, &incy);
}
inline void axpy(int n, std::complex<double> alpha, const std::complex<double>* x, int incx,
                 std::complex<double>* y, int incy) {
  zaxpy_(&n, &alpha, x, &incx, y, &incy);
}

inline float dot(int n, const float* x, int incx, const float* y, int incy) {
  return sdot_(&n, x, &incx, y, &incy);
}
inline double dot(int n, const double* x, int incx, const double* y, int incy) {
  return ddot_(&n, x, &incx, y, &incy);
}

#endif  // BLAS1_H
