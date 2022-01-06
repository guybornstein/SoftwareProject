#define PY_SSIZE_T_CLEAN
#include <Python.h>


static PyObject* fit(PyObject* self, PyObject* args);


static PyObject* fit(PyObject* self, PyObject* args) {
    char *filename;
    PyArg_ParseTuple(args, "s", &filename);
    printf("%s\n", filename);
    return Py_BuildValue("s", filename);
}


static PyMethodDef capiMethods[] = {
    {
        "fit",
        (PyCFunction) fit,
        METH_VARARGS,
        PyDoc_STR("A c implementation of kmeans algorithm.")},
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduleDef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    capiMethods
};


PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    PyObject *module;
    module = PyModule_Create(&moduleDef);
    if (!module) {
        return NULL;
    }
    return module;
};