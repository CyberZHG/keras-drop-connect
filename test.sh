#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_drop_connect tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --with-doctest --cover-package=keras_drop_connect
