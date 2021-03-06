#! /usr/bin/bash

mypy . || exit 1

project_files=($(ls xidiff/*.py))
test_files=($(ls tests/*.py))
python_files=("${project_files[@]}" "${test_files[@]}")

for python_file in ${python_files[@]};
do
    echo "Linting ${python_file}"
    pylint ${python_file} || exit 1
done
