#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>

#include "symnmf.h"


static double** build_matrix_from_lists(PyObject *lst, int *n, int *m) {
    /* Build a C matrix from a list passed from python */
    PyObject* item_lst;
    PyObject* item;
    PyObject* index;
    double** A;
    int i, j;

    /* Get length of list */
    *n = PyObject_Length(lst);
    /* Allocate matrix rows */
    A = (double**)malloc(*n * sizeof(double*));
    if (A == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }
    
    /* Load matrix values from lst */
    for (i = 0; i < *n; i++) {
        index = PyLong_FromLong(i);
        item_lst = PyObject_GetItem(lst, index);

        /* Get length of row */
        *m = PyObject_Length(item_lst);
        /* Allocate row */
        A[i] = (double*)malloc(*m * sizeof(double));
        if (A[i] == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
            return NULL;
        }
        
        /* Load values from lst to row */
        for (j = 0; j < *m; j++) {
            index = PyLong_FromLong(j);
            item = PyObject_GetItem(item_lst, index);
            A[i][j] = PyFloat_AsDouble(item);
        }
    }
    return A;
}


static PyObject* build_lists_from_matrix(double** A, int n, int m) {
    /* Build a lst to pass to python from C matrix of size n x m */
    PyObject* lists;
    PyObject* lst;
    PyObject* item;
    int i, j;

    lists = PyList_New(n);

    /* For row of matrix */
    for (i = 0; i < n; i++) {
        lst = PyList_New(m);

        /* For value of row */
        for (j = 0; j < m; j++) {
            item = Py_BuildValue("d", A[i][j]);
            PyList_SetItem(lst, j, item);
        }

        PyList_SetItem(lists, i, lst);
    }

    return lists;
}


static PyObject* sym(PyObject *self, PyObject *args) {
    /* C module function to call sym_c */
    double** X;
    PyObject* X_lst;
    PyObject* lists;
    int n, d;

    /* Get 2D list from python */
    if (!PyArg_ParseTuple(args, "O", &X_lst)) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    /* Make C matrix from python list */
    X = build_matrix_from_lists(X_lst, &n, &d);

    /* Call sym_c function */
    double** result = sym_c(X, n, d);

    /* Build python-passable list from result */
    lists = build_lists_from_matrix(result, n, n);

    /* Free memory */
    free_matrix(X, n);
    free_matrix(result, n);

    return lists;
}


static PyObject* ddg(PyObject *self, PyObject *args) {
    /* C module function to call ddg_c */
    double** X;
    PyObject* X_lst;
    PyObject* lists;
    int n, d;

    /* Get 2D list from python */
    if (!PyArg_ParseTuple(args, "O", &X_lst)) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    /* Make C matrix from python list */
    X = build_matrix_from_lists(X_lst, &n, &d);

    /* Call ddg_c function */
    double** result = ddg_c(X, n, d);

    /* Build python-passable list from result */
    lists = build_lists_from_matrix(result, n, n);

    /* Free memory */
    free_matrix(X, n);
    free_matrix(result, n);

    return lists;
}


static PyObject* norm(PyObject *self, PyObject *args) {
    /* C module function to call norm_c */
    double** X;
    PyObject* X_lst;
    PyObject* lists;
    int n, d;

    /* Get 2D list from python */
    if (!PyArg_ParseTuple(args, "O", &X_lst)) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    /* Make C matrix from python list */
    X = build_matrix_from_lists(X_lst, &n, &d);

    /* Call norm_c function */
    double** result = norm_c(X, n, d);

    /* Build python-passable list from result */
    lists = build_lists_from_matrix(result, n, n);

    /* Free memory */
    free_matrix(X, n);
    free_matrix(result, n);

    return lists;
}


static PyObject* symnmf(PyObject *self, PyObject *args) {
    /* C module function to call symnmf_c */
    double** H_0;
    double** W;
    double** result;
    PyObject* H_0_lst;
    PyObject* W_lst;
    PyObject* lists;
    int n, k;

    /* Get two 2D lists from python */
    if (!PyArg_ParseTuple(args, "OO", &H_0_lst, &W_lst)) {
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    /* Make C matrices from python lists */
    H_0 = build_matrix_from_lists(H_0_lst, &n, &k);
    W = build_matrix_from_lists(W_lst, &n, &n);
    
    /* Call symnmf_c function */
    result = symnmf_c(H_0, W, n, k);
    
    /* Build python-passable list from result */
    lists = build_lists_from_matrix(result, n, k);
    
    /* Free memory */
    free_matrix(H_0, n);
    free_matrix(W, n);
    free_matrix(result, n);

    return lists;
}


static PyMethodDef symnmfMethods[] = {
    {"sym",
        (PyCFunction)sym,
        METH_VARARGS,
        PyDoc_STR("C module function to call sym_c")},
    {"ddg",
        (PyCFunction)ddg,
        METH_VARARGS,
        PyDoc_STR("C module function to call ddg_c")},
    {"norm",
        (PyCFunction)norm,
        METH_VARARGS,
        PyDoc_STR("C module function to call norm_c")},
    {"symnmf",
        (PyCFunction)symnmf,
        METH_VARARGS,
        PyDoc_STR("C module function to call symnmf_c")},
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