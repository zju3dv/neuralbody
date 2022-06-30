cd spconv
git checkout abf0acf30f5526ea93e687e3f424f62d9cd8313a
git submodule update --init --recursive
python setup.py bdist_wheel
pip install dist/spconv-1.2.1-cp38-cp38-linux_x86_64.whl
