import time
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from data.process import pre_prpcess, post_process


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(
            binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size,
                          bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


if __name__ == "__main__":
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)

    ENGINE_PATH = 'weights/alexnet.engine'
    img_path = r"/media/ps/244e88e1-d2e1-477f-9e37-7b9cb43b842a/LB/code/mmpretrain/data/dog/test/n02089867/n0208986700000003.jpg"

    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    numpy_im = pre_prpcess(img_path)

    inputs, outputs, bindings, stream = allocate_buffers(engine)
    inputs[0].host = numpy_im

    start = time.time()
    for i in range(1000):
        trt_outputs = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        label, prob = post_process(trt_outputs[0])

    end = time.time()
    print("cost:{:.3}s pred class:{} pred prob:{:.3}".format(
        (end-start), label, prob))
    print("total cost:{:.3}s".format((end-start)))
