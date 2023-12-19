NVCC = nvcc
NVFLAGS = -diag-suppress=177 -arch=sm_89

TARGET = gemm

SRCS = gemm1.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVFLAGS) -o $@ $^

ptx: $(SRCS)
	$(NVCC) $(NVFLAGS) -DTO_PTX --ptx -o $(TARGET).ptx $^

test: $(SRCS)
	$(NVCC) $(NVFLAGS) -DTEST -o $@ $^
	./$@
ncu: $(SRCS)
	$(NVCC) $(NVFLAGS) $^
	./nncu.sh $(v)
	rm a.out

clean:
	rm -f $(TARGET) test 