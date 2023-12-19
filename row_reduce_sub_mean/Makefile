NVCC = nvcc
NVFLAGS = -diag-suppress=177 -arch=sm_89

TARGET = row_reduce_sub_mean

SRCS = row_reduce_sub_mean.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVFLAGS) $^

ptx: $(SRCS)
	$(NVCC) $(NVFLAGS) -DTO_PTX --ptx -o $(TARGET).ptx $^

test: $(SRCS)
	$(NVCC) $(NVFLAGS) -DTEST $^
	./a.out
	
ncu: $(SRCS)
	$(NVCC) $(NVFLAGS) $^
	./nncu.sh $(v)

clean:
	rm -f $(TARGET) test 