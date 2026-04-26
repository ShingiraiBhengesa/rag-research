# Source before running any attack/* code on gpu03.
# Fixes libstdc++ CXXABI_1.3.15 errors when conda libs need newer libicu/sqlite3.
export LD_LIBRARY_PATH=/home/sbhengesa/.conda/envs/adaptiverag/lib:$LD_LIBRARY_PATH
