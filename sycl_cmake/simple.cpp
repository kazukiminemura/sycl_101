#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl; // (optional) avoids need for "sycl::" before SYCL names

int main()
{
    int data[1024]; // Allocates data to be worked on
    queue myQueue;  // Create default queue to enqueue work
    // By wrapping all the SYCL work in a {} block, we ensure all
    // SYCL tasks must complete before exiting the block,
    // because the destructor of resultBuf will wait.
    { // Wrap our data variable in a buffer.
        buffer<int, 1> resultBuf{data, range<1>{1024}};
        // Create a command group to issue commands to the queue.
        myQueue.submit([&](handler &cgh)
        {
            // Request access to the buffer without initialization
            accessor writeResult{resultBuf, cgh, write_only, no_init};
            // Enqueue a parallel_for task with 1024 work-items.
            cgh.parallel_for(1024, [=](auto idx)
            {
                // Initialize each buffer element with its own rank number starting at 0
                writeResult[idx] = idx;
            }); // End of the kernel function
        });                        // End of the queue commands
    } // End of scope, so wait for the queued work to complete
    // Print result
    for (int i = 0; i < 1024; i++)
    {
        std::cout <<''data[''<< i << ''] = '' << data[i] << std::endl;
    }
    return 0;
}