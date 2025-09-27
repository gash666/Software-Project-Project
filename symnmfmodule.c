#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>

#include "symnmf.h"


static double** build_matrix_from_lists(PyObject *lst, int *n, int *m) {
    PyObject* item_lst;
    PyObject* item;
    PyObject* index;
    double** A;
    int i, j;

    *n = PyObject_Length(lst);
    
    A = (double**)malloc(*n * sizeof(double*));
    if (A == NULL) {
        printf("An Error Has Occurred8\n");
        exit(1);
    }
    
    for (i = 0; i < *n; i++) {
        index = PyLong_FromLong(i);
        item_lst = PyObject_GetItem(lst, index);
        *m = PyObject_Length(item_lst);
        
        Py_DECREF(index);
        
        A[i] = (double*)malloc(*m * sizeof(double));
        if (A[i] == NULL) {
            printf("An Error Has Occurred9\n");
            exit(1);
        }
        
        for (j = 0; j < *m; j++) {
            index = PyLong_FromLong(j);
            item = PyObject_GetItem(item_lst, index);
            A[i][j] = PyFloat_AsDouble(item);
            
            Py_DECREF(item);
            Py_DECREF(index);
        }
        
        Py_DECREF(item_lst);
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
    PyObject* X_lst;
    PyObject* lists;
    int n, d;

    if (!PyArg_ParseTuple(args, "O", &X_lst)) {
        printf("An Error Has Occurred10\n");
        exit(1);
    }

    X = build_matrix_from_lists(X_lst, &n, &d);

    double** result = sym_c(X, n, d);

    lists = build_lists_from_matrix(result, n, n);

    free_matrix(X, n);
    free_matrix(result, n);

    return lists;
}


static PyObject* ddg(PyObject *self, PyObject *args) {
    double** X;
    PyObject* X_lst;
    PyObject* lists;
    int n, d;

    if (!PyArg_ParseTuple(args, "O", &X_lst)) {
        printf("An Error Has Occurred11\n");
        exit(1);
    }

    X = build_matrix_from_lists(X_lst, &n, &d);

    double** result = ddg_c(X, n, d);

    lists = build_lists_from_matrix(result, n, n);

    free_matrix(X, n);
    free_matrix(result, n);

    return lists;
}


static PyObject* norm(PyObject *self, PyObject *args) {
    double** X;
    PyObject* X_lst;
    PyObject* lists;
    int n, d;

    if (!PyArg_ParseTuple(args, "O", &X_lst)) {
        printf("An Error Has Occurred12\n");
        exit(1);
    }

    X = build_matrix_from_lists(X_lst, &n, &d);

    double** result = norm_c(X, n, d);

    lists = build_lists_from_matrix(result, n, n);

    free_matrix(X, n);
    free_matrix(result, n);

    return lists;
}


static PyObject* symnmf(PyObject *self, PyObject *args) {
    double** H_0;
    double** W;
    double** result;
    PyObject* H_0_lst;
    PyObject* W_lst;
    PyObject* lists;
    int n, k;

    if (!PyArg_ParseTuple(args, "OO", &H_0_lst, &W_lst)) {
        printf("An Error Has Occurred13\n");
        exit(1);
    }

    H_0 = build_matrix_from_lists(H_0_lst, &n, &k);
    W = build_matrix_from_lists(W_lst, &n, &n);
    
    result = symnmf_c(H_0, W, n, k);
    
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
    "symnmf_module",
    NULL,
    -1,
    symnmfMethods
};


PyMODINIT_FUNC PyInit_symnmf_module(void) {
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m) {
        return NULL;
    }
    return m;
}