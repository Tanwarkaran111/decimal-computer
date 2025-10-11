#include <Python.h>

/* Delegate initializer: provide PyInit_gemm_native by calling PyInit_gemm */
extern PyObject *PyInit_gemm(void);
PyMODINIT_FUNC PyInit_gemm_native(void) { return PyInit_gemm(); }
