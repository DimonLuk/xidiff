#! /bin/bash


pip-compile --generate-hashes --output-file=requirements/lock/main.lock requirements/main.txt
pip-compile --generate-hashes --output-file=requirements/lock/development.lock requirements/development.txt