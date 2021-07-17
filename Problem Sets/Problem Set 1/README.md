 没什么特别的，2d * 2d的grid/block

```c++
int blockId = blockIdx.x + blockIdx.y * gridDim.x;
int threadId = blockId * (blockDim.x * blockDim.y)
 + (threadIdx.y * blockDim.x) + threadIdx.x;
```





测试结果：

![](out.jpg)


来点不一样的：
![](grey_shoot.jpg)

