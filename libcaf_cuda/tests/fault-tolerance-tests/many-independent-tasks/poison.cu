extern "C" __global__ void poison_kernel(int* dummy) {
    // Perform a dummy operation
    *dummy = 0;
    // Intentionally trigger a memory fault (Illegal Address)
    int* p = (int*)0;
    *p = 42;
}
