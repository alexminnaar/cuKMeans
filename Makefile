NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall
main.exe: cukmeans.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@