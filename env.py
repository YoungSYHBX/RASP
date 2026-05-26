import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch Git Version: {torch.version.git_version}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Built with CUDA: {torch.version.cuda}")
print(f"Built with cuDNN: {torch.backends.cudnn.enabled}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print("--- CUDA Device Info ---")
    print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    print("------------------------")
else:
    print("--- Potential Issues ---")
    # 检查是否有常见的 CUDA 环境问题线索
    import os
    print(f"CUDA_HOME Environment Variable: {os.environ.get('CUDA_HOME', 'Not Set')}") # 检查 CUDA_HOME (如果设置了的话)
    print(f"CUDA_PATH Environment Variable: {os.environ.get('CUDA_PATH', 'Not Set')}") # 检查 CUDA_PATH (Windows)
    print(f"PATH contains 'CUDA': {'CUDA' in os.environ.get('PATH', '').upper()}") # 检查 PATH 中是否包含 CUDA 相关路径
    try:
        import ctypes
        ctypes.CDLL("nvcuda.dll") # Windows
        print("nvcuda.dll loaded successfully (Windows)")
    except OSError as e:
        print(f"Failed to load nvcuda.dll (Windows): {e}")

    # 尝试加载 CUDA runtime 库 (Linux/macOS)
    try:
        import ctypes.util
        cuda_lib_path = ctypes.util.find_library("cudart") # Linux/macOS
        if cuda_lib_path:
             print(f"Found CUDA runtime library: {cuda_lib_path}")
             # Attempt to load it
             ctypes.CDLL(cuda_lib_path)
             print("CUDA runtime library loaded successfully (Linux/macOS)")
        else:
             print("Could not find CUDA runtime library (cudart) in standard paths (Linux/macOS)")
    except OSError as e:
        print(f"Failed to load CUDA runtime library (Linux/macOS): {e}")
    except AttributeError:
        print("ctypes.util.find_library not available on this platform (likely Windows)")

    print("--------------------------")