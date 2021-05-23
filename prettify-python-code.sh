#! /usr/bin/bash

project_files=($(ls xidiff/*.py))
test_files=($(ls tests/*.py))
python_files=("${project_files[@]}" "${test_files[@]}")

for python_file in ${python_files[@]};
do
    echo "Prettifying ${python_file}"
    # sort imports
    isort --atomic ${python_file}
    # apply autopep8
    autopep8 --in-place --aggressive --aggressive --aggressive ${python_file}
done