# from tvm.script import tir as T

@T.prim_func
def main(A_handle: T.handle, B_handle: T.handle, C_handle: T.handle):
    A = T.match_buffer(A_handle, (1024, 1024), "float16", strides=(1024, 1))
    B = T.match_buffer(B_handle, (1024, 512), "float16", strides=(512, 1))
    C = T.match_buffer(C_handle, (1024, 512), "float16", strides=(512, 1))
    # with T.block("root"):
    bx = T.launch_thread("blockIdx.x", 4)
    by = T.launch_thread("blockIdx.y", 8)
    tx = T.launch_thread("threadIdx.x", 128)
    ty = T.launch_thread("threadIdx.y", 1)
    tz = T.launch_thread("threadIdx.z", 1)
    with T.block("tilelang_root"):
        T.reads(A[by * 128, 0:993], B[0:1024, bx * 128:bx * 128 + 128], C[by * 128, bx * 128])
        T.writes()
        A_shared = T.alloc_buffer((128, 32), "float16", scope="shared.dyn")
        B_shared = T.alloc_buffer((32, 128), "float16", scope="shared.dyn")
        C_local = T.alloc_buffer((128, 128), scope="local.fragment")
        T.fill(T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 16384, 2), 0)
        for ko in T.serial(32, annotations={"num_stages": 3}):
            T.copy(T.region(A[by * 128, ko * 32], 1, 128, 32), T.region(A_shared[0, 0], 2, 128, 32), -1, T.bool(False), 0)
            for k in T.parallel(32):
                for j in T.parallel(128):
                    B_shared[k, j] = B[ko * 32 + k, bx * 128 + j]
            T.gemm(T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, 0, 4096, 1), T.tvm_access_ptr(T.type_annotation("float32"), C_local.data, 0, 16384, 3), T.bool(False), T.bool(False), 128, 128, 32, 0, T.bool(False), 32, 128, 0, 0, 1, 0)
        T.copy(T.region(C_local[0, 0], 1, 128, 128), T.region(C[by * 128, bx * 128], 2, 128, 128), -1, T.bool(False), 0)