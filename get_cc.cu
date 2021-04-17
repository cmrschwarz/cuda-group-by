// return the cuda compute capabiltiy for device 0 on the current hardware
// this is not just for convenience, but also directly used by the cmake build
#include <stdio.h>
#include <stdlib.h>
int main()
{
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0)) {
        fprintf(stderr, "Failed to get cuda device properties for device 0\n");
        return EXIT_FAILURE;
    }
    printf("%i\n", prop.major * 10 + prop.minor);
    return EXIT_SUCCESS;
}
