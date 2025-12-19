cmake .. -G Ninja   -DCMAKE_PREFIX_PATH="C:\Users\owner\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\share\cmake"   -DPython_ROOT_DIR="C:\Users\owner\AppData\Local\Programs\Python\Python312"   -DCMAKE_CUDA_ARCHITECTURES=86


cmake --build . -j
