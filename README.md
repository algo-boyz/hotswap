**Hot-Swap ONNX Runtime Session**

1. Versioned model pkgs in `models/...`
2. Atomic symlink flip (`current → new-version`)
3. Create **new** `Ort::Session`
4. Warm-up inference
5. Thread-safe ptr swap (mutex + `std::shared_ptr`)
6. Drop old session

Sessions are independent. The old one can be kept alive until the new one is healthy. No process restart, no TensorRT engine rebuild needed.

### Architecture

- **Double-buffer style**: inference threads always hold a `shared_ptr` to *current* session. The swap is instantaneous under a short lock.
- **Warm-up** before the swap so first inference after flip is fast.
- **External data** (`.onnx` + `.onnx.data`) is treated as one atomic package.
- Ready for small “model agent” that does download → hash/signature verify → stage → flip → signal the inference process

### How swap is safe

- Inference threads call `GetSession()` to receive a `shared_ptr` that keeps the *old* session alive until threads finish their current `Run()`
- Manager creates new session + warms it up **outside** the lock
- Only ptr swap itself is under mutex
- Once the last inference thread releases its `shared_ptr`, the old `Ort::Session` is destroyed automatically

### TODO
|         |                |
|---------|----------------|
| **Verification** | Read `manifest.json` (sha256, opset, ORT version, input/output names). Verify the tarball + signature before the flip. |
| **External data** | Treat `model.onnx` + `model.onnx.data` as one atomic unit; never flip until both are present. |
| **Memory** | For large models on Jetson, consider `session_options.DisableMemPattern()` or careful graph-opt level so load time stays acceptable. |
| **CUDA / JetPack** | Pin the exact ORT build that matches your JetPack + CUDA + cuDNN matrix. Prefer pure CUDA EP for simplest hot-swap (TensorRT EP caches engines per session). |
| **Model agent** | Small C++/gRPC process that owns download → verify → stage → symlink flip → signals the inference process (or the inference process polls the symlink). |
| **A/B system** | Keep JetPack / ORT / your binaries on the NVIDIA A/B partitions; keep only the versioned `.onnx` packages on the mutable volume. |
