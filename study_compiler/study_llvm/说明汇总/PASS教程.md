在你提供的 **HelloWorld.cpp** 和 **CMakeLists.txt** 中，一切看起来都很简单，但它可以 **成功运行**，而我之前给的一些示例代码却容易触发

```
CommandLine Error: Option 'use-dereferenceable-at-point-semantics' registered more than once!
```

这个典型的 **CommandLine 冲突**。

其实，**差异点**主要集中在 **CMake 链接策略** 和 **Pass 插件机制** 上。以下是需要重点反思的几个地方：

------

# **1. CMake 中的差异：SHARED + dynamic_lookup vs MODULE + 手动链接**

在你能成功运行的 CMakeLists.txt 中，核心写法是：

```cmake
add_library(HelloWorld SHARED HelloWorld.cpp)
target_link_libraries(HelloWorld
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
```

- **使用 SHARED 而不是 MODULE**
- **在 macOS 平台下使用 `-undefined dynamic_lookup`**；在 Linux 上默认不会手动链接 LLVM 库。

这样做的结果是：**所有 LLVM 符号都在 opt 进程里已经存在**，插件在运行时只要引用它们即可，无需再自己链接一遍。
 **如果插件重复链接 LLVMCore、LLVMPasses 等库**，则会导致同一个全局变量（例如 `CommandLine` 相关的全局注册器）被多次注册，从而触发

```
Option 'xxx' registered more than once!
```

而我之前给你的示例，往往会包含：

```cmake
add_library(MyPass MODULE src/MyPass.cpp)
target_link_libraries(MyPass PRIVATE LLVMCore LLVMPasses)
```

这会在生成的 `.so` 插件里再次“嵌入”部分 LLVM 符号，与 `opt` 中已经存在的 LLVM 符号冲突，便极容易导致 **CommandLine** 重复注册报错。

------

## **1.1 为什么这样会冲突？**

- **`opt`** 自身已经静态或动态链接了 LLVMCore/LLVMPasses 等库，**包含全局注册器**。
- **你的插件** 又通过 `target_link_libraries(... LLVMCore LLVMPasses)` 把同样的符号和全局变量链接进来，一旦和 `opt` 合并时，就会导致重复注册。

这个就是“命令行选项在动态库中被重复注册”最常见的原因。

------

# **2. 新 Pass Manager 并不需要手动链接 LLVMCore / LLVMPasses**

在 **LLVM 15+** 的新 Pass Manager 插件模式下，**opt** 已经带有所有 Pass Manager 的符号。

- 你的 `.so` **只要包含你自己的 Pass**（比如 HelloWorld 代码），对 LLVM 符号的引用会在运行时由 `opt` 提供；不需要自己链接。
- 这就是你提供的 CMake 示例里做的事情：**完全依赖 “-undefined dynamic_lookup”**（macOS）或 Linux 默认的 “弱符号动态链接” 方式；不会把 LLVMCore/LLVMPasses 再重复弄进插件。

------

# **3. SHARED vs MODULE 的差异**

- **`MODULE`**：CMake 会把这个库当作“**不能被其他库链接**，只能被运行时动态加载的插件”。
- **`SHARED`**：CMake 认为这是个 **通用的共享库**，其他程序可以 **链接** 它，或者动态加载它。

在 LLVM 插件场景下，两者都能用，但：

- 如果用 `MODULE` 并且你手动 `target_link_libraries(MyPass LLVMCore ...)`，就更容易冲突。
- 如果用 `SHARED` 并且只做 `-undefined dynamic_lookup`（或在 Linux 上默认不链接），就避免了重复注册。

你的示例成功的主要原因就是：**SHARED + 不主动链接任何 LLVM 库**。

------

# **4. 总结：代码到底“错”在哪？**

简单说：

> **我的示例**（或者很多 LLVM 15+ 教程示例）为了“看上去更明确”，往往用了
>
> ```cmake
> add_library(MyPass MODULE MyPass.cpp)
> target_link_libraries(MyPass PRIVATE LLVMCore LLVMPasses)
> ```
>
> 结果却给插件打包进了一部分 LLVM 符号，**与 opt 里的全局注册器冲突**。

而 **你的示例** 则是：

> ```cmake
> add_library(HelloWorld SHARED HelloWorld.cpp)
> target_link_libraries(HelloWorld "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
> ```
>
> **不主动链接**任何 LLVM 库，符号引用全靠 `opt` 的宿主进程来满足，避免重复的全局注册。

所以，**并不是 LLVM Pass 代码本身写错**，而是 **CMake 链接策略** 不同导致冲突。

------

# **5. 正确编译新 Pass Manager 插件的两种方式**

### **5.1 不链接 LLVMCore/LLVMPasses**

- **只编译你的 Pass 源文件**，用 `SHARED` 或 `MODULE` 均可，但**不要** `target_link_libraries(MyPass LLVMCore LLVMPasses)`。
- 在 macOS 上用 `-undefined dynamic_lookup`；在 Linux 上往往可行，只要不强制链接就行。

### **5.2 链接，但要确保“无重复”**

如果你确实想 `target_link_libraries(MyPass LLVMCore ...)`，那就必须保证 `opt` 没有把这些库静态链接进去（比较麻烦），或者你使用“标记符号为可合并”之类的方案——通常很不值得。

------

# **6. 结论**

> 你给我的就一直报 `CommandLine Error`，而你给我的“HelloWorld”却没问题，**本质原因**是：
>
> - **你的 CMake 没有重复链接 LLVM**
> - **我（或者示例）给的写法又链接了 LLVMCore/LLVMPasses**，让插件和 `opt` 重复注册全局命令行选项

**这并不是 Pass 代码本身的逻辑错误，而是 CMake 链接策略导致的全局符号冲突**。

**只要改成和你的一样**：**`SHARED + 不要链接 LLVMCore/LLVMPasses`** 或使用 “`-undefined dynamic_lookup`” 方式，就能避免 `CommandLine` 重复注册的错误。

------

## **再次感谢你的指点**

“对比学习”后，最根本的区别就是**对 LLVM 符号链接的处理**。希望这些解释能帮你理解为什么 “我的示例会冲突，而你的示例却能成功”！



以下是一份 **全新编写的 LLVM Pass 教程**，以 **HelloWorld** 为例，演示如何在 **LLVM 19+** 下编写并使用 **New Pass Manager** 插件，同时避免常见的 `CommandLine` 符号冲突问题。

------

# **🔹 教程概览**

1. [环境准备](https://chatgpt.com/c/67c58999-383c-8013-b8ed-9a950e3dc93b#环境准备)
2. [工程目录结构](https://chatgpt.com/c/67c58999-383c-8013-b8ed-9a950e3dc93b#工程目录结构)
3. [编写 HelloWorld Pass (HelloWorld.cpp)](https://chatgpt.com/c/67c58999-383c-8013-b8ed-9a950e3dc93b#编写-helloworld-pass-helloworldcpp)
4. [编写 CMakeLists.txt](https://chatgpt.com/c/67c58999-383c-8013-b8ed-9a950e3dc93b#编写-cmakeliststxt)
5. [编译并运行](https://chatgpt.com/c/67c58999-383c-8013-b8ed-9a950e3dc93b#编译并运行)
6. [为什么不会触发 CommandLine 冲突](https://chatgpt.com/c/67c58999-383c-8013-b8ed-9a950e3dc93b#为什么不会触发-commandline-冲突)

------

# **1. 环境准备**

1. **安装 LLVM 19+**（如 LLVM 19 或 LLVM 21 都可以）

   - APT 安装 (如系统提供)：

     ```bash
     sudo apt update
     sudo apt install llvm-19 clang-19 lld-19 lldb-19 llvm-19-dev
     ```

   - 或 [手动编译安装 LLVM](https://llvm.org/docs/GettingStarted.html)（略）。

2. **确认 LLVM 工具版本一致**

   ```bash
   llvm-config --version
   opt --version
   clang --version
   ```

   如果都输出 `19.0.0`（或对应的版本号），即说明安装成功且一致。

3. **CMake**

   ```bash
   sudo apt install cmake
   cmake --version  # 例如 3.20+
   ```

------

# **2. 工程目录结构**

准备一个最小工程，名为 `hello-pass`：

```
hello-pass/
 ├── src/
 │   └── HelloWorld.cpp
 └── CMakeLists.txt
```

------

# **3. 编写 HelloWorld Pass (HelloWorld.cpp)**

**新 Pass Manager** 插件模式，核心思路是：

- 定义一个结构体 `HelloWorld`，继承 `PassInfoMixin<HelloWorld>`
- 实现 `run(Function &F, FunctionAnalysisManager &)`
- 使用 `llvmGetPassPluginInfo()` 注册给 `opt`

编写 `src/HelloWorld.cpp`：

```cpp
//-----------------------------------------------------------------------------
// FILE: HelloWorld.cpp
//
// A simple LLVM pass that visits all functions in a module and prints
// their names and argument counts.
//
// Usage (New PM):
//   opt -load-pass-plugin=./build/HelloWorld.so -passes="hello-world" \
//       -disable-output test.ll
//-----------------------------------------------------------------------------

#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

// This is our pass. It just prints function name + arg count.
struct HelloWorld : public PassInfoMixin<HelloWorld> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    errs() << "(HelloWorld) Function: " << F.getName() << "\n";
    errs() << "(HelloWorld)   #Args: " << F.arg_size() << "\n";
    return PreservedAnalyses::all(); // We don't modify the IR
  }

  // 允许该 pass 在所有函数（即使有 `optnone` 标记）中运行
  static bool isRequired() { return true; }
};

} // end anonymous namespace

//-----------------------------------------------------------------------------
// Pass Registration (New PM)
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getHelloWorldPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "HelloWorld", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "hello-world") {
            FPM.addPass(HelloWorld());
            return true;
          }
          return false;
        }
      );
    }
  };
}

//-----------------------------------------------------------------------------
// This is the core interface for pass plugins
//-----------------------------------------------------------------------------
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getHelloWorldPluginInfo();
}
```

**要点**：

1. **`PassInfoMixin<HelloWorld>` + `run(Function &F, ...)`**：New Pass Manager 风格
2. **`llvmGetPassPluginInfo()`**：让 `opt` 能识别并加载本插件
3. **无 `#include "llvm/Support/CommandLine.h"`** → 避免全局命令行冲突

------

# **4. 编写 CMakeLists.txt**

在 `hello-pass/` 目录下创建 `CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.20)
project(hello-pass)

# 1) 设置 LLVM 安装路径 (可在命令行 -DLT_LLVM_INSTALL_DIR=xxx 指定)
set(LT_LLVM_INSTALL_DIR "" CACHE PATH "LLVM installation directory")

# 2) 让 CMake 查找 LLVMConfig.cmake
list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm")

# 3) 查找 LLVM
find_package(LLVM CONFIG REQUIRED)
if("${LLVM_VERSION_MAJOR}" VERSION_LESS 19)
  message(FATAL_ERROR "Need LLVM 19 or newer, found ${LLVM_VERSION_MAJOR}")
endif()

# 4) 设置编译选项，与 LLVM 配置保持一致
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

# 5) 构建 HelloWorld 插件
add_library(HelloWorld SHARED src/HelloWorld.cpp)

# 6) 不链接 LLVMCore/LLVMPasses, 避免 CommandLine 冲突
# 只在 macOS 上用 -undefined dynamic_lookup
target_link_libraries(HelloWorld
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)
```

**解释**：

- **第 5 步**：用 `SHARED`，不使用 `MODULE`，因为我们希望在 Linux 上不主动链接 LLVM 库，这样就不会重复注册符号
- **第 6 步**：不做 `target_link_libraries(HelloWorld PRIVATE LLVMCore LLVMPasses)`，以免产生重复符号
- **`"$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"`**：在 macOS 上允许“弱链接”；在 Linux 上会自动弱链接

------

# **5. 编译并运行**

## **5.1 编译**

1. 指定 LLVM 安装路径

    (如 

   ```
   /usr/lib/llvm-19
   ```

   )

   ```bash
   cd hello-pass
   rm -rf build
   mkdir build && cd build
   cmake -DLT_LLVM_INSTALL_DIR=/usr/lib/llvm-19 ..
   make -j$(nproc)
   ```

2. 检查生成的插件

   ```bash
   ls -lh HelloWorld.so
   ```

   如果看到 

   ```
   HelloWorld.so
   ```

   ，说明编译成功。

------

## **5.2 测试 Pass**

1. 准备测试文件 `test.c`

   ```c
   #include <stdio.h>
   void foo(int x) { printf("x=%d\n", x); }
   int main() {
     foo(42);
     return 0;
   }
   ```

2. 编译为 IR

   ```bash
   clang -S -emit-llvm test.c -o test.ll
   ```

3. 运行 Pass

   ```bash
   opt -load-pass-plugin ./HelloWorld.so -passes="hello-world" -disable-output test.ll
   ```

   输出示例

   ：

   ```
   (HelloWorld) Function: foo
   (HelloWorld)   #Args: 1
   (HelloWorld) Function: main
   (HelloWorld)   #Args: 0
   ```

   表示 Pass 成功运行。

------

# **6. 为什么不会触发 CommandLine 冲突？**

许多教程里会写：

```cmake
add_library(MyPass MODULE MyPass.cpp)
target_link_libraries(MyPass PRIVATE LLVMCore LLVMPasses)
```

这种做法会把 **`LLVMCore`**、**`LLVMPasses`** 库链接进插件 `.so` 文件。再加上 **`opt`** 本身早已包含了这些全局符号，导致 **全局命令行选项重复注册**。

而我们这里：

1. **用 `SHARED` 且不显式链接任何 LLVM 库**
2. 在 Linux 上默认 weak link 到 `opt` 里已经存在的 LLVM 符号
3. 在 macOS 上通过 `-undefined dynamic_lookup` 实现相同效果

这样就**不会把 LLVM 全局变量打包进插件**，从而**避免了 “Option 'xxx' registered more than once!”** 的冲突。

------

# **🔹 总结**

1. **使用 New Pass Manager**：`PassInfoMixin` + `llvmGetPassPluginInfo()`
2. **CMake 不要主动链接 LLVMCore/LLVMPasses**
3. **用 SHARED 并让 `opt` 提供 LLVM 符号**
4. **避免全局 CommandLine 冲突**

只要按本教程编写，你的 LLVM Pass 插件就可以在 **LLVM 19+** 下**无命令行冲突**地运行！

------

## **额外提示**

- 如果你想要修改 IR（而不仅仅是打印），照样在 `run()` 方法里更改即可。只需返回 `PreservedAnalyses::none()`（表示 IR 被修改）。
- 如果以后想在 CMake 中使用更复杂的操作，比如链接外部库，需确保**别把 LLVMCore/LLVMPasses 等重复链接**。
- 如果再次遇到 `CommandLine Error: Option 'xxx' registered more than once`，通常是因为**你重复链接了 LLVM 库**或**加载了多个包含同样符号的插件**。

希望这个“零冲突”的示例教程能帮你理解 **如何正确编写 LLVM Pass 插件**，并 **避免命令行冲突**，祝学习愉快！



下面我将逐步解释这两条 CMake 命令的语法和作用，并介绍如何从零开始编写一个简单的 CMakeLists.txt 文件。

------

## 一、target_link_libraries 命令概述

`target_link_libraries` 命令用于指定一个目标程序（如可执行文件或库）在链接时需要链接哪些库。这条命令不仅告诉编译器要将哪些库合并到目标中，还能设置“使用要求”（usage requirements），控制这些库的链接信息是否传递给依赖此目标的其它目标。

CMake 提供三种主要的使用要求关键字：

- **PRIVATE**：仅在构建当前目标时使用，链接信息不会传递给依赖此目标的其它目标。
- **PUBLIC**：既在构建当前目标时使用，同时传递给依赖此目标的其它目标。
- **INTERFACE**：只传递给依赖此目标的其它目标，不用于当前目标的构建。

------

## 二、解释示例 1

```cmake
target_link_libraries(MyPass PRIVATE LLVMCore LLVMPasses)
```

### 作用和语法解析

1. **MyPass**
    这是你定义的目标名称，通常在前面通过 `add_library(MyPass ...)` 或 `add_executable(MyPass ...)` 创建。
2. **PRIVATE**
    关键字说明下面列出的库（LLVMCore 和 LLVMPasses）只在构建 MyPass 时使用，MyPass 的使用者（例如链接 MyPass 的程序）不需要再链接这些库。
3. **LLVMCore 和 LLVMPasses**
    这两个是 LLVM 提供的库目标，包含了 LLVM 核心功能和与 Pass 管理相关的功能。当你在你的插件中调用 LLVM 的 API 时，这些库提供了必须的符号和实现。

### 背后的思考

- 如果你用 **PRIVATE**，说明你只需要在编译你的 MyPass 时链接这些库，而不希望把它们“打包”到最终目标中供其它依赖者使用。
- 在某些场景下，链接 LLVM 库可能会造成全局符号重复（例如 opt 本身已经链接了这些库），因此有时会选择不链接或采取特殊措施（如在 macOS 上用 `-undefined dynamic_lookup`）。

------

## 三、解释示例 2

```cmake
target_link_libraries(HelloWorld
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)
```

### 作用和语法解析

1. **HelloWorld**
    这是你的另一个目标名称，通常通过 `add_library(HelloWorld ...)` 创建。
2. **"$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"**
    这里使用了 **生成器表达式**（Generator Expression）。生成器表达式以 `$<...>` 形式书写，其作用是在生成构建文件时根据条件产生不同的字符串内容。
   - **`$<PLATFORM_ID:Darwin>`**：这是一个条件表达式，判断当前编译平台是否是 Darwin（macOS 的内核）。
   - 如果条件成立（即在 macOS 平台上），则生成 `-undefined dynamic_lookup` 这个字符串；如果条件不成立，则生成空字符串。
   - **`-undefined dynamic_lookup`** 是一个链接器选项，它告诉 macOS 的链接器允许目标中的符号在加载时再解析（即允许存在未定义符号，因为它们会在运行时由宿主程序提供）。这样可以避免在编译插件时由于缺少某些符号而报错。

### 背后的思考

- 这种写法可以在不同平台上产生不同的链接器选项。在 macOS 上，使用 `-undefined dynamic_lookup`；而在其他平台（如 Linux）上，则不添加这个选项。
- 这种方法通常用于编译插件，确保插件在加载时能够正确解析符号，不会因静态链接冲突而引发错误。

------

## 四、从 0 开始编写 CMakeLists.txt 的入门讲解

### 1. **基本框架**

每个 CMake 工程一般从下面几行开始：

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyProjectName)
```

- **cmake_minimum_required(VERSION 3.20)**：指定使用的最低 CMake 版本。
- **project(MyProjectName)**：定义项目名称。

### 2. **加载外部库或配置**

如果你的项目依赖外部库（例如 LLVM），可以用 `find_package` 查找它：

```cmake
# 假设你要使用 LLVM，且需要指定 LLVM 安装目录
set(LT_LLVM_INSTALL_DIR "/usr/lib/llvm-19" CACHE PATH "LLVM installation directory")
list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")
find_package(LLVM CONFIG REQUIRED)
```

- `set(LT_LLVM_INSTALL_DIR ...)` 定义一个缓存变量，允许用户通过命令行指定 LLVM 安装目录。
- `list(APPEND CMAKE_PREFIX_PATH ...)` 将 LLVM 的 CMake 配置目录添加到 CMake 搜索路径中。
- `find_package(LLVM CONFIG REQUIRED)` 查找 LLVM，并且如果找不到则终止构建。

### 3. **设置编译选项**

通常需要设置 C++ 标准和一些编译选项：

```cmake
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
```

- 设置使用 C++ 17 标准。
- 根据 LLVM 的设置（通常 LLVM 禁用 RTTI），设置编译器标志。

### 4. **添加源文件生成目标**

假设你要生成一个动态链接库（插件）：

```cmake
add_library(HelloWorld SHARED src/HelloWorld.cpp)
```

- **add_library(HelloWorld SHARED src/HelloWorld.cpp)**：创建一个名为 `HelloWorld` 的共享库，源文件在 `src/HelloWorld.cpp`。

### 5. **链接库**

如前面讨论，你可以选择不链接 LLVM 库（避免重复注册），或者根据需要链接：

- 如果你希望不重复链接 LLVM 库（推荐插件场景下）：

```cmake
target_link_libraries(HelloWorld
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)
```

- 如果你需要链接其他库：

```cmake
target_link_libraries(MyPass PRIVATE LLVMCore LLVMPasses)
```

注意：链接 LLVMCore/LLVMPasses 会把 LLVM 的符号拉入插件中，可能导致 opt 中已注册的符号重复，所以插件开发时一般不直接链接（尤其在 Linux 平台上）。

### 6. **完整示例**

下面是一个完整的 CMakeLists.txt 示例（适用于 LLVM 插件项目）：

```cmake
cmake_minimum_required(VERSION 3.20)
project(HelloWorldPlugin)

# 1. 设置 LLVM 安装目录（请根据实际情况修改）
set(LT_LLVM_INSTALL_DIR "/usr/lib/llvm-19" CACHE PATH "LLVM installation directory")

# 2. 添加 LLVM 配置路径
list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")

# 3. 查找 LLVM
find_package(LLVM CONFIG REQUIRED)
if("${LLVM_VERSION_MAJOR}" VERSION_LESS 19)
  message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 19 or above")
endif()

# 4. 包含 LLVM 头文件
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

# 5. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

# 6. 添加目标，注意这里我们使用 SHARED 并且不链接 LLVM 库
add_library(HelloWorld SHARED src/HelloWorld.cpp)

# 7. 在 macOS 上允许未定义符号
target_link_libraries(HelloWorld
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)
```

------

## 五、总结

- **target_link_libraries(MyPass PRIVATE LLVMCore LLVMPasses)**
  - 将目标 `MyPass` 链接到 LLVM 提供的库 `LLVMCore` 和 `LLVMPasses`。使用 `PRIVATE` 表示这些库只用于构建该目标，不会传递给依赖者。
  - 这种写法可能会导致符号重复注册（如果 opt 已经链接了这些库），因此在插件开发中不常用。
- **target_link_libraries(HelloWorld "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")**
  - 利用生成器表达式，在 macOS 平台上添加链接器选项 `-undefined dynamic_lookup`，允许插件中未定义的符号在运行时由主程序提供。
  - 在非 macOS 平台上，不会添加这个选项，从而避免链接器错误。
- **CMakeLists.txt 从 0 开始写：**
  - 定义 CMake 最低版本和项目名称
  - 设置和查找依赖（如 LLVM）
  - 配置编译选项（C++ 标准、编译器标志）
  - 添加目标（库或可执行文件）
  - 使用 target_link_libraries 指定链接库和平台相关链接器选项

希望这个详细讲解能帮助你理解这两部分语法以及如何编写 CMakeLists.txt。如果还有疑问，欢迎继续提问！



Below is a detailed, line-by-line explanation of the provided HelloWorld.cpp code for an LLVM pass using the new Pass Manager. This explanation will help you understand each part of the program at a fine granularity.

------

```cpp
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
```

- Header Inclusions:
  - `llvm/Passes/PassPlugin.h`: Provides the interface and utilities needed to create a pass plugin that can be loaded by LLVM’s `opt` tool.
  - `llvm/Passes/PassBuilder.h`: Contains definitions for the new Pass Manager and tools to register passes.
  - `llvm/IR/PassManager.h`: Defines the classes for managing passes over the LLVM Intermediate Representation (IR), such as `FunctionPassManager`.
  - `llvm/Support/raw_ostream.h`: Provides the `errs()` stream, used to print error messages or debug output. It’s similar to C++’s `std::cerr`.

------

```cpp
using namespace llvm;
```

- Namespace Usage:
  - This allows you to use LLVM classes (e.g., `Function`, `errs()`, etc.) without having to prefix them with `llvm::` each time.

------

### **Anonymous Namespace**

```cpp
namespace {
```

- Anonymous Namespace:
  - Placing the pass implementation inside an anonymous namespace limits its scope to the current translation unit. This helps avoid symbol conflicts with other parts of the program or other plugins.

------

### **Defining the Pass**

```cpp
struct HelloWorld : public PassInfoMixin<HelloWorld> {
```

- Structure Declaration:
  - Here we declare a struct named `HelloWorld` that inherits from `PassInfoMixin<HelloWorld>`.
  - The `PassInfoMixin` is a CRTP (Curiously Recurring Template Pattern) base class provided by LLVM that helps with some boilerplate for pass registration and identification.

------

```cpp
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
```

- **run() Method:**
  - This is the main entry point of the pass when it is run on each function in the IR.
  - It takes two parameters:
    - `Function &F`: A reference to the current function that the pass is analyzing.
    - `FunctionAnalysisManager &`: A reference to a manager that provides analysis results for the function. (In this simple pass, we don't use it.)
- **Return Type:**
  - `PreservedAnalyses`: This indicates which analyses are preserved (i.e., remain valid) after running this pass. In our case, we don’t modify the function, so we preserve all analyses.

------

```cpp
    errs() << "(HelloWorld) Function: " << F.getName() << "\n";
    errs() << "(HelloWorld)   #Args: " << F.arg_size() << "\n";
```

- Printing Debug Information:
  - `errs()` is LLVM’s error/debug output stream (similar to `std::cerr`).
  - The first line prints the name of the function (`F.getName()`).
  - The second line prints the number of arguments in the function (`F.arg_size()`).
  - This is the core functionality of this pass: it simply reports some details about each function it visits.

------

```cpp
    return PreservedAnalyses::all();
  }
};
```

- **Preservation of Analyses:**
  - Since this pass only prints information and does not modify the function’s IR, it returns `PreservedAnalyses::all()` to indicate that all previously computed analyses remain valid.
- **End of the Struct:**
  - This completes the definition of the `HelloWorld` pass.

------

```cpp
} // end anonymous namespace
```

- Closing the Anonymous Namespace:
  - Ends the anonymous namespace. Everything defined within this namespace is local to this source file.

------

### **Pass Plugin Registration**

```cpp
llvm::PassPluginLibraryInfo getHelloWorldPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "HelloWorld", LLVM_VERSION_STRING,
```

- getHelloWorldPluginInfo() Function:
  - This function returns a `PassPluginLibraryInfo` structure that contains metadata about the pass plugin.
  - `LLVM_PLUGIN_API_VERSION` and `LLVM_VERSION_STRING` are macros provided by LLVM that ensure compatibility.
  - The string `"HelloWorld"` is the name of the plugin (used for identification).

------

```cpp
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "hello-world") {
                    FPM.addPass(HelloWorld());
                    return true;
                  }
                  return false;
                });
          }};
```

- Lambda Function for Pass Registration:

  - The lambda function passed to the `PassPluginLibraryInfo` constructor registers a callback with the `PassBuilder`.

  - ```
    PB.registerPipelineParsingCallback(...)
    ```

    :

    - This callback is used by the LLVM `opt` tool to add your pass to the pass pipeline when the pass name `"hello-world"` is encountered.

  - Inside the Callback:

    - It checks if the provided `Name` equals `"hello-world"`.
    - If it does, it calls `FPM.addPass(HelloWorld())` to add an instance of your pass to the current `FunctionPassManager`.
    - Returning `true` indicates that the pass was successfully registered.

------

### **Extern "C" and Final Export**

```cpp
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getHelloWorldPluginInfo();
}
```

- **Extern "C":**
  - This block declares an external C function (`llvmGetPassPluginInfo`) so that the LLVM `opt` tool can find and load your plugin.
  - Using `extern "C"` disables C++ name mangling, which is necessary for dynamic loading.
- **LLVM_ATTRIBUTE_WEAK:**
  - This attribute marks the function as weakly defined, which allows for flexible symbol resolution during dynamic linking.
- **Return Value:**
  - The function simply calls and returns the result of `getHelloWorldPluginInfo()`, making your pass plugin available to the LLVM pass infrastructure.

------

## **Summary**

- **Header Files:** Bring in LLVM’s Pass Plugin and IR manipulation facilities.
- **Anonymous Namespace:** Encapsulates your pass to avoid name collisions.
- **Pass Definition (`HelloWorld`):** Inherits from `PassInfoMixin` and implements the `run()` method for the new Pass Manager, printing function names and argument counts.
- **Pass Registration:** Uses a lambda inside `getHelloWorldPluginInfo()` to register the pass under the name `"hello-world"`, allowing it to be invoked via the `opt` tool.
- **Extern "C" Export:** Exports the registration function using C linkage so that LLVM's dynamic loader can find it.

This explanation should give you a fine-grained understanding of how the LLVM pass works and how each part of the code contributes to the overall functionality. If you have any further questions about specific parts or need additional clarification, please let me know!





That's great to hear! Now that you have mastered the basics by coding a simple HelloWorld pass, here are some suggestions and topics to further your study of LLVM:

------

### 1. **Deepen Your Understanding of LLVM IR**

- **Learn the LLVM IR Structure:**
   Study the design of LLVM IR, its instruction set, data types, and the SSA form. The [LLVM Language Reference Manual](https://llvm.org/docs/LangRef.html) is an excellent resource.
- **Examine Existing IR Code:**
   Compile some simple C/C++ programs using Clang with `-emit-llvm` and analyze the resulting `.ll` files. This helps you understand how high-level constructs map to IR.

------

### 2. **Explore Advanced Passes**

- **Analysis Passes:**
   Learn how to write passes that gather information without modifying the IR. Examples include control flow graph analysis, data flow analysis, and dependency analysis.
- **Transformation Passes:**
   Experiment with optimization passes that transform IR. For example, try implementing your own version of constant propagation, dead code elimination, or loop unrolling.
- **New Pass Manager vs. Legacy Pass Manager:**
   Understand the differences between the new and legacy pass managers. Dive into how the new pass manager handles analysis preservation, pass dependencies, and pipeline registration.

------

### 3. **Work with LLVM Libraries**

- **LLVM Core Libraries:**
   Get familiar with libraries like `LLVMCore`, `LLVMPasses`, and `LLVMAnalysis`. Explore how they are used to implement transformations and analyses.
- **Linker and Code Generation:**
   Look into how LLVM’s backend works. Study the code generation process, target-specific optimizations, and how LLVM generates machine code from IR.

------

### 4. **Study and Contribute to LLVM**

- **LLVM Documentation and Tutorials:**
   Read the [LLVM Developer’s Handbook](https://llvm.org/docs/DeveloperPolicy.html) and [Getting Started Guide](https://llvm.org/docs/GettingStarted.html) for deeper insights.
- **Follow the LLVM Community:**
   Engage with the LLVM community by joining mailing lists, reading blog posts, and exploring open-source projects that extend LLVM.
- **Contribute to LLVM:**
   Start with small contributions like fixing bugs or adding features in less critical components. This will help you understand LLVM’s coding practices and architecture.

------

### 5. **Hands-on Projects and Further Reading**

- **Implement More Passes:**
   Create passes that perform non-trivial optimizations or analyses. For example, try writing a pass that inlines small functions or one that performs aggressive constant propagation.
- **Use LLVM in Your Projects:**
   Consider integrating LLVM into a larger project or using it as a backend for a custom language. This practical experience will reinforce your learning.
- **Recommended Books and Resources:**
  - *"LLVM Essentials"* – A book that introduces LLVM concepts and practical applications.
  - *"Getting Started with LLVM Core Libraries"* – Focuses on using LLVM as a library to build custom compilers.
  - Online courses and tutorials, such as those on [Udemy](https://www.udemy.com/) or [Coursera](https://www.coursera.org/), sometimes offer courses related to compiler construction with LLVM.

------

By exploring these topics, you'll gain a deeper understanding of LLVM's architecture and its powerful optimization capabilities. Feel free to ask if you need more detailed guidance on any of these areas!

## OpcodeCounter CMakeLists.txt explanation

Below is a detailed, line-by-line explanation of the provided CMakeLists.txt file, breaking down each section and command:

------

```cmake
cmake_minimum_required(VERSION 3.20)
```

- **Purpose:**
   This command specifies the minimum version of CMake required to process the project.
- **Detail:**
   By requiring version 3.20 or higher, you ensure that all the CMake features used in this file are available. If a user has an older CMake version, an error will be issued.

------

```cmake
project(llvm-tutor-opcode-counter)
```

- **Purpose:**
   Defines the name of the project.
- **Detail:**
   The project is named "llvm-tutor-opcode-counter", which helps identify it in generated build files and logs. It also sets up some default variables (like `PROJECT_SOURCE_DIR`).

------

### **Section 1: Load LLVM Configuration**

```cmake
# Set this to a valid LLVM installation dir
set(LT_LLVM_INSTALL_DIR "" CACHE PATH "LLVM installation directory")
```

- **Purpose:**
   This line defines a cache variable named `LT_LLVM_INSTALL_DIR` with an empty default value.
- Detail:
  - The variable type is `PATH`, indicating that it is expected to be a directory path.
  - The CACHE keyword means its value is stored in CMakeCache.txt, allowing users to override it from the command line (e.g., `-DLT_LLVM_INSTALL_DIR=/path/to/llvm`).
  - The description "LLVM installation directory" is shown in CMake GUIs to help users understand what value to provide.

------

```cmake
# Add the location of LLVMConfig.cmake to CMake search paths (so that find_package can locate it)
list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")
```

- **Purpose:**
   Adds a directory to the CMake search path used by `find_package()`.
- Detail:
  - `CMAKE_PREFIX_PATH` is a list of directories that CMake searches for package configuration files.
  - By appending `"${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/"`, you tell CMake where to look for LLVM’s configuration file (`LLVMConfig.cmake`).
  - This makes it possible to locate LLVM even if it’s installed in a non-standard location.

------

```cmake
find_package(LLVM CONFIG)
```

- **Purpose:**
   Locates and loads the LLVM package using its configuration files.
- Detail:
  - The `CONFIG` option tells CMake to use package configuration files (like LLVMConfig.cmake) rather than using a Find module.
  - If LLVM is not found, this command will fail unless you set it to optional. In our case, it is required for the project.

------

```cmake
if("${LLVM_VERSION_MAJOR}" VERSION_LESS 19)
  message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 19 or above")
endif()
```

- **Purpose:**
   Ensures that the found LLVM version meets the minimum required version (19 or above).
- Detail:
  - `${LLVM_VERSION_MAJOR}` is a variable provided by LLVM’s configuration files that indicates the major version.
  - If the condition evaluates to true (i.e., LLVM's major version is less than 19), the script aborts with a fatal error message.

------

```cmake
# Add LLVM include directories and your own header directory
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} inc)
```

- **Purpose:**
   Sets up include directories for the project.
- Detail:
  - `${LLVM_INCLUDE_DIRS}` contains the LLVM header paths found by `find_package(LLVM CONFIG)`.
  - The `inc` directory is added as an additional include directory for your project’s headers.
  - The `SYSTEM` keyword tells the compiler that these are system headers, which can suppress warnings for them.

------

### **Section 2: Build Configuration**

```cmake
# Use the same C++ standard as LLVM does
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
```

- **Purpose:**
   Specifies that the project should be compiled using C++17.
- Detail:
  - `CMAKE_CXX_STANDARD` is set to 17, ensuring compatibility with LLVM, which typically uses C++17.
  - The CACHE keyword allows this setting to be visible and changeable in CMake GUIs.

------

```cmake
# LLVM is normally built without RTTI. Be consistent with that.
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()
```

- **Purpose:**
   Ensures that if LLVM was built without RTTI (Run-Time Type Information), your project is compiled similarly.
- Detail:
  - `LLVM_ENABLE_RTTI` is usually defined by LLVM’s configuration; if it is not enabled, the compiler flag `-fno-rtti` is added.
  - This maintains consistency between your project and LLVM’s build configuration, which can prevent compatibility issues.

------

### **Section 3: Add the Target**

```cmake
add_library(HelloWorld SHARED src/HelloWorld.cpp)
```

- **Purpose:**
   Creates a shared library target named `HelloWorld`.
- Detail:
  - `add_library` is used to build a library.
  - `SHARED` specifies that the library is dynamic (i.e., a `.so` file on Linux or a `.dylib` on macOS).
  - The source file is specified as `src/HelloWorld.cpp`. (In your case, you mentioned you want to change this to `src/OpcodeCounter.cpp`, so you would replace the file name accordingly.)

------

```cmake
target_link_libraries(HelloWorld
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)
```

- **Purpose:**
   Specifies linker options for the `HelloWorld` target.

- Detail:

  - `target_link_libraries` adds libraries or linker flags to the target.

  - The argument here is a 

    generator expression

    :

    - `$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>`
    - This means: if the platform ID equals `Darwin` (i.e., if you're on macOS), then add the linker flag `-undefined dynamic_lookup`.

  - Why this flag?

    - On macOS, this flag tells the linker to allow undefined symbols in the shared library. This is useful for LLVM pass plugins because the host application (`opt`) already provides the necessary LLVM symbols.
    - On other platforms (like Linux), this generator expression evaluates to an empty string, so no extra linker flag is added.

------

## **Summary**

- **cmake_minimum_required** and **project**: Establish the CMake version and project name.

- Set LLVM configuration variables

  :

  - `LT_LLVM_INSTALL_DIR` is defined to let the user specify where LLVM is installed.
  - CMake’s prefix path is appended to find LLVM’s config files.

- **find_package(LLVM CONFIG)**: Locates LLVM using the configuration files.

- **Version Check**: Ensures that LLVM’s version is at least 19.

- **Include Directories**: Adds LLVM’s include directories and the local `inc` directory for header files.

- **C++ Standard and Flags**: Sets the project to use C++17 and, if necessary, disables RTTI to be consistent with LLVM.

- **Target Creation**: Creates a shared library target (`HelloWorld`) using the source file.

- **Linker Options**: Uses a generator expression to add macOS-specific linker flags to allow dynamic symbol lookup.

This step-by-step explanation should help you understand each component of the CMakeLists.txt and how they work together to configure and build your LLVM plugin. If you need further clarification on any part, please feel free to ask!
