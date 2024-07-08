#pragma once

#ifndef PYTHON_ENVIRONMENT_H
#define PYTHON_ENVIRONMENT_H

#include <vector>
#include <string>
#include <stdexcept>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

class PythonEnvironment {
public:
    PythonEnvironment();
    ~PythonEnvironment();
};

std::vector<float> process_python_array(const std::vector<float>& input_vector, const std::string& python_script);
std::string process_token_ids(const std::vector<int64_t>& token_ids, const std::string& python_script);

#endif // PYTHON_ENVIRONMENT_H
