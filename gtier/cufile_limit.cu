// Where does cuFile actually stop?  Registering device buffers is how a client
// would hold a large staging window with GDS, so if that has a hard cap it is a
// structural limit rather than a tuning one.  Tested directly, not through our
// benchmark, so the answer does not depend on our code.
#include <cufile.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
    const char *path = argv[1];
    size_t chunk = (argc > 2 ? atoll(argv[2]) : 64) << 20;
    double cap_gib = argc > 3 ? atof(argv[3]) : 40.0;

    if (cuFileDriverOpen().err != CU_FILE_SUCCESS)
        std::printf("driver open failed (compat mode)\n");

    int fd = open(path, O_RDONLY | O_DIRECT);
    CUfileDescr_t d{}; d.handle.fd = fd; d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t h;
    if (cuFileHandleRegister(&h, &d).err != CU_FILE_SUCCESS) {
        std::printf("handle register failed\n"); return 1;
    }

    std::vector<void *> bufs;
    double total = 0;
    int n = 0;
    for (;;) {
        void *p = nullptr;
        if (cudaMalloc(&p, chunk) != cudaSuccess) {
            std::printf("cudaMalloc failed at %.1f GiB (%d buffers)\n", total, n);
            break;
        }
        CUfileError_t e = cuFileBufRegister(p, chunk, 0);
        if (e.err != CU_FILE_SUCCESS) {
            std::printf("cuFileBufRegister FAILED at %.1f GiB (%d buffers), err=%d\n",
                        total, n, e.err);
            cudaFree(p);
            break;
        }
        bufs.push_back(p);
        total += chunk / 1073741824.0;
        ++n;
        if (n % 64 == 0) std::printf("  registered %4d buffers = %6.1f GiB\n", n, total);
        if (total >= cap_gib) { std::printf("reached cap %.1f GiB, stopping\n", cap_gib); break; }
    }
    std::printf("RESULT: cuFile registered %.1f GiB in %d x %zu MiB buffers\n",
                total, n, chunk >> 20);

    if (!bufs.empty()) {
        ssize_t r = cuFileRead(h, bufs[0], chunk, 0, 0);
        std::printf("  read into first buffer: %zd bytes\n", r);
    }
    for (void *p : bufs) { cuFileBufDeregister(p); cudaFree(p); }
    cuFileHandleDeregister(h); cuFileDriverClose(); close(fd);
    return 0;
}
