## **📌 Affine 和 Memory Reference（MemRef）优化解析**

在 MLIR 和 LLVM 编译优化中，**Affine 优化** 和 **Memory Reference（MemRef）优化** 是**两个核心优化技术**，它们主要用于： ✅ **提升循环性能**（Loop Optimizations）
 ✅ **减少内存访问延迟**（Memory Access Optimization）
 ✅ **提高缓存命中率**（Cache Locality）
 ✅ **为并行计算做准备**（Parallelization & Vectorization）

------

# **🌟 1. 什么是 Affine（仿射）优化？**

### **🔹 1.1 Affine 变换的定义**

**Affine 变换** 是指 **线性变换 + 平移**，用于优化循环和数据访问模式。
 在编译优化中，**Affine 优化主要用于循环变换（Loop Transformations）**，比如：

- **Loop Tiling（循环切块）**
- **Loop Unrolling（循环展开）**
- **Loop Fusion（循环融合）**
- **Loop Interchange（循环交换）**
- **Dependence Analysis（循环依赖分析）**

------

### **🔹 1.2 具体 Affine 优化示例**

#### **✅ 示例 1：普通的 MLIR 循环**

```mlir
func.func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      %a = load %A[%i, %j]
      %b = load %B[%i, %j]
      %c = load %C[%i, %j]
      %res = arith.addf %a, %b
      store %res, %C[%i, %j]
    }
  }
}
```

**这个代码有什么问题？**

- **没有优化**，每次迭代都会访问 `A[i, j]` 和 `B[i, j]`，造成**大量的缓存未命中（Cache Miss）**。
- **可以通过 Affine 变换优化！**

------

### **✅ 示例 2：使用 Affine Loop Tiling（循环切块）优化**

```mlir
affine.for %ii = 0 to 1024 step 32 {
  affine.for %jj = 0 to 1024 step 32 {
    affine.for %i = ii to ii + 32 {
      affine.for %j = jj to jj + 32 {
        %a = load %A[%i, %j]
        %b = load %B[%i, %j]
        %c = load %C[%i, %j]
        %res = arith.addf %a, %b
        store %res, %C[%i, %j]
      }
    }
  }
}
```

**优化点：** ✅ **Tiling 使得计算更集中，每次处理 32x32 的小块数据，提高缓存命中率！**
 ✅ **减少 DRAM 访问，提高计算吞吐量！**
 ✅ **更适合 SIMD（AVX, NEON）和 GPU 计算！**

------

### **🔹 1.3 Affine 优化如何应用**

#### **MLIR 提供 Affine Pass 进行优化**

```bash
mlir-opt input.mlir --affine-loop-tile="tile-size=32"
```

✅ **这样可以自动应用 Loop Tiling！**

#### **更多 Affine 变换**

| **优化方式**          | **作用**                           |
| --------------------- | ---------------------------------- |
| **Loop Unrolling**    | 减少循环控制开销，提高指令级并行度 |
| **Loop Fusion**       | 组合多个循环，减少内存访问         |
| **Loop Interchange**  | 交换循环层次，提高缓存友好性       |
| **Affine Scheduling** | 重新排列循环，提高并行性           |

------

## **🌟 2. Memory Reference（MemRef）优化**

### **🔹 2.1 什么是 MemRef（Memory Reference）？**

在 MLIR 中，**MemRef（Memory Reference）表示内存缓冲区（类似于 LLVM IR 的 `memref`）**。
 ✅ **MemRef 允许静态/动态内存分配，并提供优化支持！**
 ✅ **优化 Memory Reference 可以减少不必要的内存访问，提升数据局部性！**

------

### **🔹 2.2 MemRef 访问优化**

#### **✅ 示例 1：非优化的 MemRef 访问**

```mlir
func.func @memref_example(%A: memref<1024xf32>) {
  affine.for %i = 0 to 1024 {
    %a = load %A[%i]   // 直接加载 A[i]，每次访问都会去内存
    %b = arith.addf %a, %a
    store %b, %A[%i]
  }
}
```

🔴 **问题：**

- 每次迭代都会从内存加载 `A[i]`，即使 `A[i]` 之前已经加载过！
- 造成 **多次 DRAM 访问，影响缓存命中率**。

------

#### **✅ 示例 2：MemRef Cache 优化**

```mlir
func.func @memref_example(%A: memref<1024xf32>) {
  affine.for %i = 0 to 1024 {
    %a = load %A[%i]   // 加载 A[i] 到寄存器
    affine.for %j = 0 to 8 {
      %b = arith.addf %a, %a  // 只用寄存器计算
    }
    store %b, %A[%i]   // 计算完再写回
  }
}
```

✅ **优化点：**

- **将 A[i] 读入寄存器，在循环 `j` 内复用**（减少内存访问）。
- **减少 DRAM 访问，提高缓存命中率**。

------

### **🔹 2.3 MemRef Hoisting（MemRef 提升）**

MLIR 提供 **MemRef Hoisting（内存缓冲区提升）**，可以减少不必要的 `alloc` 和 `dealloc`。

#### **✅ 示例 1：未经优化的 MemRef 代码**

```mlir
func.func @example() {
  affine.for %i = 0 to 1024 {
    %A = memref.alloc() : memref<128xf32>
    %B = memref.alloc() : memref<128xf32>
    call @compute(%A, %B)
    memref.dealloc %A
    memref.dealloc %B
  }
}
```

🔴 **问题：**

- **每次迭代都会 `alloc` 和 `dealloc`**，增加 **内存管理开销**！
- **造成额外的内存碎片**！

------

#### **✅ 示例 2：优化后的 MemRef Hoisting**

```mlir
func.func @example() {
  %A = memref.alloc() : memref<128xf32>
  %B = memref.alloc() : memref<128xf32>
  affine.for %i = 0 to 1024 {
    call @compute(%A, %B)
  }
  memref.dealloc %A
  memref.dealloc %B
}
```

✅ **优化点：**

- **`A` 和 `B` 只 `alloc` 一次，并在整个循环中复用！**
- **减少 `alloc/dealloc` 频率，提升性能！**

------

## **🌟 3. 结论**

### **🔹 Affine 变换**

✅ 主要用于 **循环优化**，包括：

- **Loop Tiling**（减少 Cache Miss）
- **Loop Unrolling**（提高并行计算）
- **Loop Fusion**（减少内存访问）
- **Loop Interchange**（提高缓存利用率）

✅ **指令：**

```bash
mlir-opt input.mlir --affine-loop-tile="tile-size=32"
```

------

### **🔹 MemRef 变换**

✅ 主要用于 **内存优化**，包括：

- **减少不必要的内存访问**
- **提高缓存命中率**
- **MemRef Hoisting（减少 alloc/dealloc 开销）**

✅ **指令：**

```bash
mlir-opt input.mlir --memref-expand
```

------

## **🚀 总结**

✅ **Affine 优化 = 循环优化（Loop Transformations）**
 ✅ **MemRef 优化 = 内存优化（Memory Reference Optimization）**
 ✅ **两者结合可以极大提高计算性能，特别是 AI、GPU、HPC 领域！** 🚀

## **📌 编译优化新手入门 Tutor**

作为新手，学习编译优化可以从 **基础理论** → **实际编译** → **优化实践** 三个阶段逐步掌握。
 本教程会一步步带你学习 **MLIR / LLVM 编译优化**，并附带**实操代码**。

------

# **🌟 1. 理解编译器工作流程**

### **📌 编译器的基本流程**

编译器将 **源代码（C/C++/Python）转换为可执行程序**，主要经过以下阶段：

```
前端（Front-end）  →  中端（Middle-end）  →  后端（Back-end）
```

| **阶段**               | **作用**              | **示例**            |
| ---------------------- | --------------------- | ------------------- |
| **前端（Front-end）**  | 解析代码并生成 IR     | 词法分析、语法分析  |
| **中端（Middle-end）** | 进行优化，生成低级 IR | 代码优化、Loop 优化 |
| **后端（Back-end）**   | 生成目标机器代码      | LLVM IR → 汇编代码  |

💡 **示例：Clang 编译 C 代码**

```bash
clang -S -emit-llvm example.c -o example.ll
```

这会输出 LLVM IR（中端 IR）。

------

# **🌟 2. 搭建编译环境**

你需要安装 **LLVM & MLIR**，建议使用最新版本：

### **📌 安装 LLVM 和 MLIR**

```bash
# Ubuntu
sudo apt install llvm clang lld
```

**或者手动编译（推荐）**

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir;clang" -DCMAKE_BUILD_TYPE=Release
ninja
```

------

# **🌟 3. 运行第一个 LLVM IR**

### **📌 1️⃣ 编写简单的 C 代码**

创建 `example.c`：

```c
int add(int a, int b) {
    return a + b;
}
```

### **📌 2️⃣ 生成 LLVM IR**

```bash
clang -S -emit-llvm example.c -o example.ll
```

查看 `example.ll`：

```llvm
define i32 @add(i32 %a, i32 %b) {
  %1 = add i32 %a, %b
  ret i32 %1
}
```

✅ 你已经成功生成了中间代码 **LLVM IR**！

------

# **🌟 4. LLVM IR 优化**

LLVM 提供 `opt` 工具用于优化：

```bash
opt -mem2reg -loop-unroll -inline example.ll -o optimized.ll
```

| **优化 Pass**      | **作用**                                |
| ------------------ | --------------------------------------- |
| **`-mem2reg`**     | 提升变量到寄存器（消除冗余 load/store） |
| **`-loop-unroll`** | 展开循环，减少分支预测                  |
| **`-inline`**      | 函数内联，减少调用开销                  |

------

# **🌟 5. MLIR 入门**

MLIR 是 **LLVM 的多层次 IR**，适用于**AI、GPU、FPGA** 等领域。

### **📌 1️⃣ 运行 MLIR**

创建 `example.mlir`：

```mlir
func @add(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32
  return %sum
}
```

运行 MLIR：

```bash
mlir-opt example.mlir --canonicalize
```

✅ MLIR 支持 **更高级的优化**！

------

# **🌟 6. MLIR 优化 Pass**

MLIR 也有很多优化 Pass，比如：

```bash
mlir-opt input.mlir --affine-loop-unroll --loop-invariant-code-motion
```

| **优化 Pass**                      | **作用**           |
| ---------------------------------- | ------------------ |
| **`--affine-loop-unroll`**         | 展开循环，减少跳转 |
| **`--loop-invariant-code-motion`** | 提取循环不变代码   |

✅ 你可以通过 **MLIR Pass** 进行高级优化！

------

# **🌟 7. MLIR 生成 LLVM IR**

```bash
mlir-opt input.mlir --convert-scf-to-llvm | mlir-translate --mlir-to-llvmir
```

✅ 这样 MLIR 代码可以转换成 **LLVM IR**，再进一步编译！

------

# **🌟 8. 高级优化**

### **📌 1️⃣ 使用 GPU 进行优化**

```bash
mlir-opt input.mlir --convert-linalg-to-gpu
```

✅ 这样可以**让代码在 GPU 上运行！**

### **📌 2️⃣ 使用 MLIR 进行 AI 加速**

MLIR 适用于 TensorFlow、PyTorch：

```bash
mlir-opt input.mlir --convert-tensor-to-linalg
```

✅ 这样可以**优化 AI 计算！**

------

# **🌟 9. 进阶学习路线**

| **阶段**   | **目标**                      | **推荐工具**               |
| ---------- | ----------------------------- | -------------------------- |
| **初学者** | 了解 LLVM IR 和 MLIR 基础     | `clang`, `mlir-opt`, `opt` |
| **进阶**   | 研究 LLVM Pass & MLIR Dialect | `mlir-opt --pass`          |
| **专家**   | 设计自定义 MLIR Pass          | `LLVM PassBuilder API`     |

------

# **🌟 10. 总结**

✅ **学习编译优化先理解 LLVM IR 和 MLIR**
 ✅ **使用 LLVM `opt` 进行优化**
 ✅ **使用 MLIR 进行高级优化（AI / GPU / 矩阵计算）**
 ✅ **掌握 `mlir-opt` 让你的编译器更智能！** 🚀

💡 **如果你是初学者，从简单的 C → LLVM IR → MLIR Pass 开始学习！**