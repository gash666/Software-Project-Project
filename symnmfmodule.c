#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>

#include "symnmf.h"


static double** build_matrix_from_lists(PyObject *args, int *n, int *m) {
    PyObject* lst;
    PyObject* item_lst;
    PyObject* item;
    double** A;
    int i, j;

    if (!PyArg_ParseTuple(args, "O", lst)) {
        return NULL;
    }

    *n = PyObject_Length(lst);

    A = (double**)malloc(*n * sizeof(double*));
    if (A == NULL) {
        return NULL;
    }

    for (i = 0; i < *n; i++) {
        item_lst = PyObject_GetItem(lst, i);
        *m = PyObject_Length(item_lst);
        
        A[i] = (double*)malloc(*m * sizeof(double));
        if (A[i] == NULL) {
            return NULL;
        }

        for (j = 0; j < *m; j++) {
            item = PyObject_GetItem(item_lst, j);
            A[i][j] = PyFloat_AsDouble(item);
        }
    }

    return A;
}


static PyObject* build_lists_from_matrix(double** A, int n, int m) {
    PyObject* lists;
    PyObject* lst;
    PyObject* item;
    int i, j;

    lists = PyList_New(n);

    for (i = 0; i < n; i++) {
        lst = PyList_New(m);

        for (j = 0; j < m; j++) {
            item = Py_BuildValue("d", A[i][j]);
            PyList_SetItem(lst, j, item);
        }

        PyList_SetItem(lists, i, lst);
    }

    return lists;
}


static PyObject* sym(PyObject *self, PyObject *args) {
    double** X;
    PyObject* lists;
    int n, d;

    X = build_matrix_from_lists(args, &n, &d);

    double** result = sym_c(X, n, d);

    lists = build_lists_from_matrix(result, n, n);

    free_matrix(X, n);
    free_matrix(result, n);

    return lists;
}


static PyObject* ddg(PyObject *self, PyObject *args) {
    double** X;
    PyObject* lists;
    int n, d;

    X = build_matrix_from_lists(args, &n, &d);

    double** result = ddg_c(X, n, d);

    lists = build_lists_from_matrix(result, n, n);

    free_matrix(X, n);
    free_matrix(result, n);

    return lists;
}


static PyObject* norm(PyObject *self, PyObject *args) {
    double** X;
    PyObject* lists;
    int n, d;

    X = build_matrix_from_lists(args, &n, &d);

    double** result = norm_c(X, n, d);

    lists = build_lists_from_matrix(result, n, n);

    free_matrix(X, n);
    free_matrix(result, n);

    return lists;
}


static PyObject* symnmf(PyObject *self, PyObject *args) {
    double** H_0;
    double** W;
    PyObject* lists;
    int n, k;

    H_0 = build_matrix_from_lists(args, &n, &k);
    W = build_matrix_from_lists(args, &n, &n);

    double** result = symnmf_c(H_0, W, n, k);

    lists = build_lists_from_matrix(result, n, k);

    free_matrix(H_0, n);
    free_matrix(W, n);
    free_matrix(result, n);

    return lists;
}


static PyMethodDef symnmfMethods[] = {
    {"sym",
        (PyCFunction)sym,
        METH_VARARGS,
        PyDoc_STR("sym")},
    {"ddg",
        (PyCFunction)ddg,
        METH_VARARGS,
        PyDoc_STR("ddg")},
    {"norm",
        (PyCFunction)norm,
        METH_VARARGS,
        PyDoc_STR("norm")},
    {"symnmf",
        (PyCFunction)symnmf,
        METH_VARARGS,
        PyDoc_STR("symnmf")},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf_c",
    NULL,
    -1,
    symnmfMethods
};


PyMODINIT_FUCN PyInit_symnmf_c(void) {
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m) {
        return NULL;
    }
    return m;
}