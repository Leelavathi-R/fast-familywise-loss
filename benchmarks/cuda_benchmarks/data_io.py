import numpy as np
import torch
import csv 
import matplotlib.pyplot as plt

FIELDNAMES = ["K", "CE", "FW_naive", "FW_topk", "FW_CUDA"]

def generate_data(batch_size, num_classes, scaling_factor = 1.0, device = 'cpu'):
    inputs = scaling_factor * torch.randn(
        batch_size, num_classes,
        device=device,
        dtype=torch.float32
        
    )
    inputs.requires_grad_(True)
    targets = torch.randint(
        0, num_classes,
        (batch_size,),
        device=device
    )
    
    return inputs, targets

def generate_data_laplace(batch_size, num_classes, scaling_factor=1.0, device='cpu'):

    scale = scaling_factor / (2.0 ** 0.5)
    dist = torch.distributions.Laplace(
        loc=torch.tensor(0.0, device=device),
        scale=torch.tensor(scale, device=device)
    )

    inputs = dist.sample((batch_size, num_classes)).float()
    inputs.requires_grad_(True)

    targets = torch.randint(
        0,
        num_classes,
        (batch_size,),
        device=device
    )

    return inputs, targets

def generate_data_gaussian_mixture( batch_size, num_classes, sigma=1.0, device='cpu' ):
    # 50/50 mixture
    choose_first = torch.rand(
        batch_size,
        num_classes,
        device=device
    ) < 0.5

    # N(0, 1)
    samples1 = torch.randn(
        batch_size,
        num_classes,
        device=device,
        dtype=torch.float32
    )

    # N(0, sigma^2)
    samples2 = sigma * torch.randn(
        batch_size,
        num_classes,
        device=device,
        dtype=torch.float32
    )

    inputs = torch.where(
        choose_first,
        samples1,
        samples2
    )

    inputs.requires_grad_(True)

    targets = torch.randint(
        0,
        num_classes,
        (batch_size,),
        device=device
    )

    return inputs, targets

def write_header(path):
    with open(path, "w", newline="") as file:
        csv.writer(file).writerow(FIELDNAMES)

def write_result(path, row):
    with open(path, "a", newline="") as file:
        csv.writer(file).writerow(row)

def plot_speed(csv_path, output_path):
    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)
        methods = [name for name in reader.fieldnames if name != "K"]
        rows = sorted(reader, key=lambda row: int(row["K"]))

    K_values = [int(row["K"]) for row in rows]

    for method in methods:
        times = [float(row[method]) for row in rows]
        if method == "CE":
            plt.plot(K_values, times, color="black", linestyle="--", linewidth=2, label=method)
        else:
            plt.plot(K_values, times, linewidth=2, label=method)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of classes (K)")
    plt.ylabel("Time (ms)")
    plt.title("Loss & grad exec time(cuda)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def write_candidate_header(path):
    with open(path, "w", newline="") as file:
        csv.writer(file).writerow(
            ["K", "candidate_mean", "candidate_percent", "reduction_percent"]
        )


def write_candidate_result(path, row):
    with open(path, "a", newline="") as file:
        csv.writer(file).writerow(row)


def plot_speed_nu_subplots(nu_values, results_dir, output_path):

    ncols = 3
    nrows = int(np.ceil(len(nu_values) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 7.5),   # larger
        sharex=True,
        sharey=True
    )

    kbar_ranges = {
        0.001: (100, 3283.91),
        0.01: (81.75, 383.47),
        0.1: (18.44, 44.94),
        1.0: (3.81, 5.68),
        10.0: (1.31, 1.56),
        100.0: (1.06, 1.03)
    }

    axes = np.atleast_1d(axes).flatten()

    for ax, nu in zip(axes, nu_values):

        csv_path = results_dir / f"speed_laplacian_{nu}.csv"

        with open(csv_path, newline="") as file:
            reader = csv.DictReader(file)

            methods = [
                name for name in reader.fieldnames
                if name != "K"
            ]

            rows = sorted(
                reader,
                key=lambda row: int(row["K"])
            )

        K_values = [
            int(row["K"])
            for row in rows
        ]

        for method in methods:

            times = [
                float(row[method])
                for row in rows
            ]

            if method == "CE":
                ax.plot(
                    K_values,
                    times,
                    color="black",
                    linestyle="--",
                    linewidth=2.2,
                    label="CE"
                )

            else:
                ax.plot(
                    K_values,
                    times,
                    linewidth=2.2,
                    label=method
                )

        ax.set_xscale("log")
        ax.set_yscale("log")

        kbar_min, kbar_max = kbar_ranges[nu]

        ax.set_title(
            rf"$\sigma={nu}$" + ", " +
            rf"Mean $\bar{{k}}$ range: {kbar_min:,}–{kbar_max:,}",
            fontsize=13,
            pad=8
        )

        ax.set_xlabel(
            "number of classes $K$",
            fontsize=13
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=12
        )

        ax.grid(
            True,
            alpha=0.3
        )

    # Remove unused subplots
    for ax in axes[len(nu_values):]:
        ax.remove()

    # Y-axis labels for left column
    for row in range(nrows):
        axes[row * ncols].set_ylabel(
            "loss + gradient time (ms)",
            fontsize=13
        )

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle(
        "Loss & grad execution time (GPU)\n"
        r"logits $\sim \mathcal{N}(0,\sigma^2)$",
        fontsize=15,
        y=0.98
    )

    # Shared legend at bottom
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.8,
        handletextpad=0.6
    )

    fig.subplots_adjust(
        top=0.82,
        bottom=0.14,
        left=0.08,
        right=0.98,
        hspace=0.50,
        wspace=0.20
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def plot_laplacian_subplots(nu_values, results_dir, output_path):

    ncols = 3
    nrows = int(np.ceil(len(nu_values) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 7.5),
        sharex=True,
        sharey=True
    )

    kbar_ranges = {
        0.001: (100, 1416.97),
        0.01: (86.66, 143.31),
        0.1: (15.09, 17.31),
        1.0: (2.50, 2.41),
        10.0: (1.13, 1.19),
        100.0: (1,1)
    }

    axes = np.atleast_1d(axes).flatten()

    for ax, sigma in zip(axes, nu_values):

        csv_path = results_dir / f"speed_laplacian_{sigma}.csv"

        with open(csv_path, newline="") as file:
            reader = csv.DictReader(file)

            methods = [
                name for name in reader.fieldnames
                if name != "K"
            ]

            rows = sorted(
                reader,
                key=lambda row: int(row["K"])
            )

        K_values = [
            int(row["K"])
            for row in rows
        ]

        for method in methods:

            times = [
                float(row[method])
                for row in rows
            ]

            if method == "CE":
                ax.plot(
                    K_values,
                    times,
                    color="black",
                    linestyle="--",
                    linewidth=2.2,
                    label="CE"
                )

            else:
                ax.plot(
                    K_values,
                    times,
                    linewidth=2.2,
                    label=method
                )

        ax.set_xscale("log")
        ax.set_yscale("log")

        kbar_min, kbar_max = kbar_ranges[sigma]

        ax.set_title(
            rf"$\sigma={sigma}$" + "\n" +
            rf"Mean $\bar{{k}}$ range: {kbar_min:,}–{kbar_max:,}",
            fontsize=13,
            pad=8
        )

        ax.set_xlabel(
            "number of classes $K$",
            fontsize=13
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=12
        )

        ax.grid(
            True,
            alpha=0.3
        )

    # Remove unused subplots
    for ax in axes[len(nu_values):]:
        ax.remove()

    # Y-axis labels for left column
    for row in range(nrows):
        axes[row * ncols].set_ylabel(
            "loss + gradient time (ms)",
            fontsize=13
        )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle(
        "Loss & grad execution time (GPU)\n"
        r"logits $\sim \mathrm{Laplace}(0,b)$, "
        r"$b=\sigma/\sqrt{2}$",
        fontsize=15,
        y=0.98
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.8,
        handletextpad=0.6
    )

    fig.subplots_adjust(
        top=0.82,
        bottom=0.14,
        left=0.08,
        right=0.98,
        hspace=0.50,
        wspace=0.20
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_mixed_subplots(sigma_values, results_dir, output_path):

    ncols = 3
    nrows = int(np.ceil(len(sigma_values) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 7.5),
        sharex=True,
        sharey=True
    )

    kbar_ranges = {
        0.001: (100, 1416.97),
        0.01: (86.66, 143.31),
        0.1: (15.09, 17.31),
        1.0: (2.50, 2.41),
        10.0: (1.13, 1.19),
        100.0: (1, 1)
    }

    axes = np.atleast_1d(axes).flatten()

    for ax, sigma in zip(axes, sigma_values):

        csv_path = results_dir / f"speed_mixed_{sigma}.csv"

        with open(csv_path, newline="") as file:
            reader = csv.DictReader(file)

            methods = [
                name for name in reader.fieldnames
                if name != "K"
            ]

            rows = sorted(
                reader,
                key=lambda row: int(row["K"])
            )

        K_values = [
            int(row["K"])
            for row in rows
        ]

        for method in methods:

            times = [
                float(row[method])
                for row in rows
            ]

            if method == "CE":
                ax.plot(
                    K_values,
                    times,
                    color="black",
                    linestyle="--",
                    linewidth=2.2,
                    label="CE"
                )
            else:
                ax.plot(
                    K_values,
                    times,
                    linewidth=2.2,
                    label=method
                )

        ax.set_xscale("log")
        ax.set_yscale("log")

        kbar_min, kbar_max = kbar_ranges[sigma]

        ax.set_title(
            rf"$\sigma={sigma}$",
            fontsize=13,
            pad=8
        )

        ax.set_xlabel(
            "number of classes $K$",
            fontsize=13
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=12
        )

        ax.grid(
            True,
            alpha=0.3
        )

    for ax in axes[len(sigma_values):]:
        ax.remove()

    # Y-axis labels for left column
    for row in range(nrows):
        axes[row * ncols].set_ylabel(
            "loss + gradient time (ms)",
            fontsize=13
        )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.suptitle(
        "Loss & grad execution time (GPU)\n"
        r"logits $\sim 0.5\,\mathcal{N}(0,1)"
        r" + 0.5\,\mathcal{N}(0,\sigma^2)$",
        fontsize=15,
        y=0.98
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.8,
        handletextpad=0.6
    )

    fig.subplots_adjust(
        top=0.82,
        bottom=0.14,
        left=0.08,
        right=0.98,
        hspace=0.50,
        wspace=0.20
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)