#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// Simple ModelManager that owns the live session and supports hot-swap
class ModelManager {
public:
    explicit ModelManager(Ort::Env& env)
        : env_(env) {}

    // Load (or hot-swap) a model version.
    // model_dir should point to something like models/perception/v1.4.3
    // which contains model.onnx (+ optional model.onnx.data) and manifest.json
    bool LoadVersion(const std::string& model_dir,
                     const std::string& /*expected_hash*/ = "") {
        const fs::path onnx_path = fs::path(model_dir) / "yolo11n-face.onnx";
        if (!fs::exists(onnx_path)) {
            std::cerr << "model not found in " << model_dir << "\n";
            return false;
        }

        // 1. Create a brand-new session (old one stays alive)
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // opts.SetIntraOpNumThreads(2);          // tune for your SoC
        // opts.EnableMemPattern();               // often good for fixed shapes

#ifdef ONNXRUNTIME_CUDA
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        // cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        opts.AppendExecutionProvider_CUDA(cuda_opts);
#endif

        std::unique_ptr<Ort::Session> new_session;
        try {
            new_session = std::make_unique<Ort::Session>(
                env_, onnx_path.c_str(), opts);
        } catch (const Ort::Exception& e) {
            std::cerr << "Failed to create session: " << e.what() << "\n";
            return false;
        }

        // 2. Warm-up (very important on Jetson / first CUDA launch)
        if (!WarmUp(*new_session)) {
            std::cerr << "Warm-up failed\n";
            return false;
        }

        // 3. Locked atomic swap
        {
            std::lock_guard<std::mutex> lock(session_mutex_);
            current_session_ = std::shared_ptr<Ort::Session>(std::move(new_session));
            current_version_  = model_dir;
        }

        std::cout << "[ModelManager] Swapped to " << model_dir << "\n";
        return true;
    }

    // Thread-safe accessor used by inference threads
    std::shared_ptr<Ort::Session> GetSession() const {
        std::lock_guard<std::mutex> lock(session_mutex_);
        return current_session_;
    }

    std::string CurrentVersion() const {
        std::lock_guard<std::mutex> lock(session_mutex_);
        return current_version_;
    }

private:
    bool WarmUp(Ort::Session& session) {
        // Minimal warm-up: allocate dummy tensors matching the model’s first input
        // and run a few inferences.  Real code should read shapes from the model
        // or from manifest.json.
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name = session.GetInputNameAllocated(0, allocator);
            auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            auto shape      = input_info.GetShape();

            // Replace dynamic dims with a realistic size for warm-up
            for (auto& d : shape) if (d < 0) d = 1;

            size_t num_elements = 1;
            for (auto d : shape) num_elements *= static_cast<size_t>(d);

            std::vector<float> dummy(num_elements, 0.0f);
            auto memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, dummy.data(), dummy.size(),
                shape.data(), shape.size());

            const char* input_names[]  = {input_name.get()};
            // We ignore outputs for warm-up
            for (int i = 0; i < 3; ++i) {
                session.Run(Ort::RunOptions{nullptr},
                            input_names, &input_tensor, 1,
                            nullptr, nullptr, 0);
            }
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "Warm-up exception: " << e.what() << "\n";
            return false;
        }
    }

    Ort::Env& env_;
    mutable std::mutex session_mutex_;
    std::shared_ptr<Ort::Session> current_session_;
    std::string current_version_;
};

bool AtomicFlipCurrent(const std::string& models_root,
                       const std::string& version) {
    // models_root/perception/current  →  models_root/perception/vX.Y.Z
    fs::path current = fs::path(models_root) / "perception" / "current";
    fs::path target  = fs::path(models_root) / "perception" / version;

    if (!fs::exists(target)) {
        std::cerr << "Target version directory missing: " << target << "\n";
        return false;
    }

    // Atomic replace of symlink (Linux)
    fs::path tmp = current.string() + ".tmp";
    std::error_code ec;
    fs::remove(tmp, ec);
    fs::create_directory_symlink(target.filename(), tmp, ec); // relative symlink
    if (ec) {
        // fallback to absolute
        fs::create_directory_symlink(target, tmp, ec);
    }
    fs::rename(tmp, current, ec);   // atomic on same filesystem
    if (ec) {
        std::cerr << "Symlink flip failed: " << ec.message() << "\n";
        return false;
    }
    return true;
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "hot-swap-demo");
    ModelManager mgr(env);
    const std::string models_root = "models";
    // models/perception/v1.4.2/model.onnx
    // models/perception/v1.5.0/model.onnx
    // models/perception/current → v1.4.2

    // Initial load via current symlink
    if (!mgr.LoadVersion(models_root + "/perception/current")) {
        std::cerr << "Initial load failed. Create a dummy model first\n";
        return 1;
    }

    // Background inference thread that continuously uses the live session
    std::atomic<bool> running{true};
    std::thread infer_thread([&] {
        while (running) {
            auto session = mgr.GetSession();   // cheap shared_ptr copy
            if (!session) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            // preprocessing + Run() here
            // For demo just print version every second
            static auto last = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            if (now - last > std::chrono::seconds(1)) {
                std::cout << "Inference using: " << mgr.CurrentVersion() << "\n";
                last = now;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Simulate model update arriving later
    std::this_thread::sleep_for(std::chrono::seconds(3));

    std::cout << "\n=== Simulating model update to v1.5.0 ===\n";
    // 1. (In real life) download + verify signature/hash into staging
    // 2. Atomic flip of the “current” symlink
    if (AtomicFlipCurrent(models_root, "v1.5.0")) {
        // 3. Hot-swap the live session
        mgr.LoadVersion(models_root + "/perception/current");
    }

    std::this_thread::sleep_for(std::chrono::seconds(5));
    running = false;
    infer_thread.join();

    std::cout << "Demo finished.\n";
    return 0;
}