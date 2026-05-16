# ONNX Export

Export trained MLP surrogates for deployment anywhere.

!!! note "MLP only"
    GP models store training data for inference and cannot be exported. Use `method="mlp"` if you need portable surrogates.

## Basic export

```python
surrogate.export("model.onnx")
# Creates: model__CL.onnx, model__CD.onnx  (one file per output)
```

## Loading with onnxruntime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model__CL.onnx")
cl = session.run(None, {"X": x_new.astype(np.float32)})[0]
```

## Deployment targets

| Target | Description |
|--------|-------------|
| Real-time control | C++ inference via ONNX Runtime C API |
| Edge devices | TensorRT-optimized ONNX on Jetson |
| Cloud serving | Triton Inference Server with ONNX backend |
| Python applications | `onnxruntime` package (pip install) |
