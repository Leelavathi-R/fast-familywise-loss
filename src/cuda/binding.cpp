#include <torch/extension.h>

std::vector<torch::Tensor> fw_radix_forward(torch::Tensor logits, torch::Tensor targets);
torch::Tensor fw_radix_backward(torch::Tensor logits, torch::Tensor targets, torch::Tensor H, torch::Tensor kappas, torch::Tensor candidate_buffer, torch::Tensor candidate_indices, torch::Tensor candidate_counts);
std::vector<torch::Tensor> fw_radix_forward_small(torch::Tensor logits,torch::Tensor targets);
torch::Tensor fw_radix_backward_small( torch::Tensor targets,torch::Tensor selected_logits, torch::Tensor kappas );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fw_forward", &fw_radix_forward, "FW Radix CUDA - FORWARD");
    m.def("fw_backward", &fw_radix_backward, "FW Radix CUDA - BACKWARD");
    m.def("fw_forward_small", &fw_radix_forward_small, "FW Radix CUDA - FORWARD");
    m.def("fw_backward_small", &fw_radix_backward_small, "FW Radix CUDA - BACKWARD");
}