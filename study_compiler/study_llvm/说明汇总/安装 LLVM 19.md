### 安装 LLVM 19

sudo apt update
sudo apt install llvm-19 clang-19 lld-19 lldb-19 llvm-19-dev

llvm-config-19 --version
opt-19 --version
clang-19 --version

sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100
sudo update-alternatives --install /usr/bin/opt opt /usr/bin/opt-19 100
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-19 100
sudo update-alternatives --install /usr/bin/llc llc /usr/bin/llc-19 100

clang --version
opt --version
llvm-config --version

sudo apt update
sudo apt install libzstd-dev

## 编译安装



git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout release/19.x
mkdir build
cd build

#-DLLVM_TARGETS_TO_BUILD="X86;ARM;NVPTX;AMDGPU"

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="clang;llvm;mlir" \
  -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

cmake --build . -- -j4
sudo cmake --install .