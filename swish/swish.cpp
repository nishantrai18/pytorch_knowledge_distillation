#include <torch/extension.h>

#include <vector>

/*
Helper utility functions
*/

/*
Derivative of sigmoid,
    s'(z) = (1 - s(z)) * s(z)
*/
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

torch::Tensor swish(torch::Tensor b, torch::Tensor z) {
  auto s = torch::sigmoid(b * z);
  return z * s;
}

/*
Derivative of swish wrt x,
    f_x'(x, b) = b * f(x, b) + sigmoid(b * x)(1 − b * f(x, b))
*/
torch::Tensor dx_swish(torch::Tensor b, torch::Tensor z) {
  auto s = torch::sigmoid(b * z);
  auto f = swish(b, z);
  return (b * f) + (s * (1 - (b * f)));
}

/*
Derivative of swish wrt beta,
    f_b'(x, b) = x * x * d_sigmoid(b * x)
*/
torch::Tensor db_swish(torch::Tensor b, torch::Tensor z) {
  auto ds = d_sigmoid(b * z);
  return z * z * ds;
}

/*
Forward and backward implementations
*/

torch::Tensor swish_forward(torch::Tensor b, torch::Tensor z) {
  return swish(b, z);
}

std::vector<torch::Tensor> swish_backward(torch::Tensor b, torch::Tensor z) {
  return {
    db_swish(b, z),
    dx_swish(b, z)
  };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &swish_forward, "Swish forward");
  m.def("backward", &swish_backward, "Swish backward");
}
