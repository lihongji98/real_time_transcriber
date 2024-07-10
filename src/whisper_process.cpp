#include "whisper_process.h"
#include <iostream>


PythonEnvironment::PythonEnvironment() {
    Py_Initialize();
    if (_import_array() < 0) {
        throw std::runtime_error("NumPy initialization failed");
    }
}

PythonEnvironment::~PythonEnvironment() {
    Py_Finalize();
}


std::vector<float> process_python_array(const std::vector<float>& input_vector, const std::string& python_script) {
    std::vector<float> result_vector;

    npy_intp dimensions[1] = {static_cast<npy_intp>(input_vector.size())};
    PyObject* py_array = PyArray_SimpleNewFromData(1, dimensions, NPY_FLOAT32, const_cast<npy_float32 *>(input_vector.data()));

    PyRun_SimpleString(
            "import sys\n"
            "class NullWriter:\n"
            "    def write(self, msg): pass\n"
            "    def flush(self): pass\n"
            "sys.stdout = NullWriter()\n"
            "sys.stderr = NullWriter()\n"
    );

    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* global_dict = PyModule_GetDict(main_module);
    PyDict_SetItemString(global_dict, "raw_audio_array", py_array);

    PyRun_SimpleString(python_script.c_str());

    PyObject* result = PyDict_GetItemString(global_dict, "processed_audio_array");

    if (PyArray_Check(result)) {
        auto* np_arr = reinterpret_cast<PyArrayObject*>(result);

        if (PyArray_TYPE(np_arr) != NPY_FLOAT32) {
            std::cerr << "Array is not of type float32" << std::endl;
        } else {
            int num_elements = PyArray_SIZE(np_arr);
            auto* data = static_cast<float*>(PyArray_DATA(np_arr));
            result_vector = std::vector<float>(data, data + num_elements);
        }
    } else {
        std::cerr << "Result is not a NumPy array" << std::endl;
    }

    Py_DECREF(py_array);

    return result_vector;
}


std::string process_token_ids(const std::vector<int64_t>& token_ids, const std::string& python_script) {
    std::string tokens;

    npy_intp dimensions[1] = {static_cast<npy_intp>(token_ids.size())};
    PyObject* token_ids_np = PyArray_SimpleNewFromData(1, dimensions, NPY_INT64, const_cast<int64_t*>(token_ids.data()));

    PyRun_SimpleString(
            "import sys\n"
            "class NullWriter:\n"
            "    def write(self, msg): pass\n"
            "    def flush(self): pass\n"
            "sys.stdout = NullWriter()\n"
            "sys.stderr = NullWriter()\n"
    );

    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* global_dict = PyModule_GetDict(main_module);
    PyDict_SetItemString(global_dict, "token_ids", token_ids_np);

    PyRun_SimpleString(python_script.c_str());

    PyObject* result = PyDict_GetItemString(global_dict, "token_string");

    // Check if the result is a string
    if (PyUnicode_Check(result)) {
        // Convert Python string to C++ string
        PyObject* result_bytes = PyUnicode_AsEncodedString(result, "utf-8", "strict");
        tokens = PyBytes_AS_STRING(result_bytes);
        Py_DECREF(result_bytes);
    } else {
        std::cerr << "Result is not a string" << std::endl;
    }


    return tokens;
}


