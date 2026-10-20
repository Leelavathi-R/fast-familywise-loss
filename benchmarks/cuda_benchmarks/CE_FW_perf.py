from pathlib import Path
import sys
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import fw_radix_cuda

import numpy as np
import statistics
from data_io import generate_data, write_header, write_result, plot_speed_nu_subplots, generate_data_laplace, plot_laplacian_subplots, generate_data_gaussian_mixture, plot_mixed_subplots
from reference_impl import FWLoss, FWLoss_topk_uniform
import time

def benchmark_speed(loss_fn, inputs, target, warmups=20, trials=200):
    def run_once():
        inputs.grad = None
        loss = loss_fn(inputs, target)
        loss.backward()

    for _ in range(warmups):
        run_once()
    
    torch.cuda.synchronize()
    times_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(trials):
        start.record()
        run_once()
        end.record()
        end.synchronize()

        times_ms.append(start.elapsed_time(end))

    return statistics.median(times_ms)                  



def loss_grad_correctness(reference_fn, test_fn, inputs, target, rtol=1e-5, atol=1e-6):
    x_ref = inputs.detach().clone().requires_grad_(True)
    x_test = inputs.detach().clone().requires_grad_(True)

    loss_ref = reference_fn(x_ref, target)
    loss_test = test_fn(x_test, target)

    torch.testing.assert_close(loss_test, loss_ref, rtol=rtol, atol=atol)

    loss_test.backward()
    loss_ref.backward()

    torch.testing.assert_close(x_test.grad, x_ref.grad, rtol=rtol, atol=atol)

def benchmark(fn, warmup=20, iterations=200):
    for _ in range(warmup):
        out = fn()

    torch.cuda.synchronize()

    times_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(iterations):
        start.record()
        out = fn()
        end.record()

        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    
    return out, statistics.median(times_ms)


def fw_radix_forward_backward(inputs, targets):

    K = inputs.size(1)

    if K < 10000:
        H, losses, mean_loss, kappas, selected_logits = fw_radix_cuda.fw_forward_small( inputs, targets ) 
        grad = fw_radix_cuda.fw_backward_small( targets, selected_logits, kappas )

    else:
        H, losses, mean_loss, kappas, candidate_buffer, candidate_indices, candidate_counts = fw_radix_cuda.fw_forward( inputs, targets ) 
        grad = fw_radix_cuda.fw_backward( inputs, targets, H, kappas, candidate_buffer, candidate_indices, candidate_counts )

    return None

def main():
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cuda"
    B = 32
    K = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    sigma = [0.001, 0.01, 0.1,1.0, 10.0, 100.0]
    benchmark_dir = Path(__file__).resolve().parent

    results_dir = benchmark_dir / "results"
    figs_dir = benchmark_dir / "figs"

    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    ce = torch.nn.CrossEntropyLoss()
    for scaling_factor in sigma:
        results_path = results_dir / f"speed_mixed_{scaling_factor}.csv"
        write_header(results_path)

        for k in K:
            
            print("---", k, "---")
            print('scaling factor:', scaling_factor)
            inputs, targets = generate_data_gaussian_mixture(B, k, scaling_factor, device)
            fw_naive = FWLoss(k, device = device)
            Hk_naive, losses_naive, mean_losse_naive, kappa_naive = fw_naive(inputs, targets)
            mean_losse_naive.backward()
            grad_naive = inputs.grad.clone()
            print("naive done")

            if k < 10000: 
                Hk_cuda, losses_cuda, mean_loss_cuda, kappa_cuda, selected_logits = fw_radix_cuda.fw_forward_small(inputs, targets) 
                grad_cuda = fw_radix_cuda.fw_backward_small(targets, selected_logits, kappa_cuda) 
            else: 
                (Hk_cuda, losses_cuda, mean_loss_cuda, kappa_cuda, candidate_buffer, candidate_indices, candidate_counts ) = fw_radix_cuda.fw_forward(inputs, targets) 
                print(
                    "Candidate counts:",
                    "min =", candidate_counts.min().item(),
                    "mean =", candidate_counts.float().mean().item(),
                    "max =", candidate_counts.max().item()
                )
                candidate_mean = ( candidate_counts.float().mean().item() ) 
                candidate_percent = ( 100.0 * candidate_mean / k ) 
                reduction_percent = ( 100.0 - candidate_percent ) 
                grad_cuda = fw_radix_cuda.fw_backward( inputs, targets, Hk_cuda, kappa_cuda, candidate_buffer, candidate_indices, candidate_counts )
            
            with torch.no_grad():
                z64 = inputs.detach().double()

                z_sorted64 = torch.sort(
                    z64, dim=1, descending=True
                ).values

                S64 = z_sorted64.cumsum(dim=1)

                k_values = torch.arange(
                    1, k + 1,
                    device=device,
                    dtype=torch.float64
                )

                H_all64 = (S64 - 1.0) / k_values

                H64, idx64 = H_all64.max(dim=1)
                kappa64 = idx64 + 1

                grad_ref = torch.zeros_like(inputs)

                for i in range(inputs.shape[0]):
                    kappa_i = int(kappa64[i].item())
                    h_i = H64[i]

                    selected = z64[i] > h_i

                    grad_ref[i, selected] = 1.0 / kappa_i
                    grad_ref[i, targets[i]] -= 1.0

                grad_ref /= inputs.shape[0]

            print("Kappa:", torch.equal(kappa_cuda.long(), kappa64.long()))
            print("kappa_cuda:", kappa_cuda)
            print("kappa_naive:", kappa64.long())

            print("Hk:", torch.allclose(Hk_naive, Hk_cuda, rtol=1e-5, atol=1e-6))
            print("Losses:",torch.allclose(losses_naive, losses_cuda, rtol=1e-5, atol=1e-6))
            print("Avg Loss:",torch.allclose(mean_losse_naive, mean_loss_cuda, rtol=1e-5, atol=1e-6))
            print("Grad:",torch.allclose(grad_ref, grad_cuda, rtol=1e-5, atol=1e-6))
            
            result_row = [k]

            ''' ----- Benchmarking Speed -----'''
            print("Evaluating Speed...")
            ce_time = benchmark_speed(lambda x, y: ce(x, y), inputs, targets)
            result_row.append(ce_time)
        
            # print(k, " ",ce_time)
            fw_naive = FWLoss(k, device = device, reduction = 'mean')
            fw_naive_time = benchmark_speed(lambda x, y: fw_naive(x, y)[2], inputs, targets)
            result_row.append(fw_naive_time)

            fw_topk = FWLoss_topk_uniform(k, device = device)
            fw_topk_time = benchmark_speed(lambda x, y: fw_topk(x, y), inputs, targets)
            result_row.append(fw_topk_time)

            _, fw_cuda_time = benchmark( lambda: fw_radix_forward_backward( inputs, targets ) )
            result_row.append(fw_cuda_time)
            write_result(results_path,result_row)

        print(f"Results saved to {results_path}")
    
    plot_path = figs_dir / "speed_mixed_all.png"
    plot_mixed_subplots(sigma, results_dir,plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()