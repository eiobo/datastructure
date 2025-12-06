# 设置CUDA编译器
NVCC := nvcc

# 编译选项
CXXFLAGS := -arch=sm_86  # 针对RTX 3060的CUDA架构

# 头文件路径
INCLUDES := -I./common

# .cu的文件名, 不包含扩展名, 每次编译修改这个变量即可
FILENAME := 4.10Warp_Shuffle_Instrcution

# 输出文件路径（不包含扩展名）
TARGETNAME := ./exefile/$(FILENAME)

# 要编译的源文件
SRC := $(FILENAME).cu ./common/common.cpp

# 编译目标
$(TARGETNAME): $(SRC)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -o $(TARGETNAME) $(SRC) 

# 运行目标：编译后自动运行程序
run: $(TARGETNAME)
	@echo
	@echo "Compilation complete. Now Running the program ..."
	@echo "-----------------------------------------------------------------------------------"
	@echo
	./$(TARGETNAME)
	@echo
	@echo "-----------------------------------------------------------------------------------"
	@echo "Program execution finished."

# 清理生成的文件（无扩展名的目标文件）
clean:
	rm -f $(TARGETNAME)