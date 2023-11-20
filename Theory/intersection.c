#include <Python.h>

int orientation(double p[2], double q[2], double r[2]) {
    double val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);
    if (val == 0) return 0;
    return (val > 0) ? 1 : 2;
}

int on_segment(double p[2], double q[2], double r[2]) {
    return (q[0] <= fmax(p[0], r[0]) && q[0] >= fmin(p[0], r[0]) &&
            q[1] <= fmax(p[1], r[1]) && q[1] >= fmin(p[1], r[1]));
}

static PyObject *do_intersect(PyObject *self, PyObject *args) {
    double p1[2], q1[2], p2[2], q2[2];

    if (!PyArg_ParseTuple(args, "(dd)(dd)(dd)(dd)", &p1[0], &p1[1], &q1[0], &q1[1], &p2[0], &p2[1], &q2[0], &q2[1])) {
        return NULL;
    }

    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    if (o1 != o2 && o3 != o4) {
        Py_RETURN_TRUE;
    }

    if (o1 == 0 && on_segment(p1, p2, q1) && (p1[0] != p2[0] || p1[1] != p2[1]) && (q1[0] != p2[0] || q1[1] != p2[1])) {
        Py_RETURN_TRUE;
    }
    if (o2 == 0 && on_segment(p1, q2, q1) && (p1[0] != q2[0] || p1[1] != q2[1]) && (q1[0] != q2[0] || q1[1] != q2[1])) {
        Py_RETURN_TRUE;
    }
    if (o3 == 0 && on_segment(p2, p1, q2) && (p2[0] != p1[0] || p2[1] != p1[1]) && (q2[0] != p1[0] || q2[1] != p1[1])) {
        Py_RETURN_TRUE;
    }
    if (o4 == 0 && on_segment(p2, q1, q2) && (p2[0] != q1[0] || p2[1] != q1[1]) && (q2[0] != q1[0] || q2[1] != q1[1])) {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}

static PyMethodDef IntersectionMethods[] = {
    {"do_intersect", do_intersect, METH_VARARGS, "Check if two line segments intersect"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef intersectionmodule = {
    PyModuleDef_HEAD_INIT,
    "intersection",
    NULL,
    -1,
    IntersectionMethods
};

PyMODINIT_FUNC PyInit_intersection(void) {
    return PyModule_Create(&intersectionmodule);
}
