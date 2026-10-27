# engine.py
"""
Polished, self-contained training engine.

Features:
- Main `train(...)` function with signature compatible with:
    results = engine.train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=5,
        device=device
    )

- Helper `train_one_epoch(...)` that shows per-batch progress with tqdm and ETA,
  and displays a dynamic loss value for each batch.

- Uses `rich` for colorful, stylized prints and timestamps.

- Contains runnable mock Model/DataLoader/Optimizer/Loss for demonstration so the file
  can be executed immediately to see the logging and progress behavior.

Notes:
- If you pass real PyTorch model/dataloaders/optimizer/loss to train(...),
  the engine will use them as-is.
"""

from __future__ import annotations
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from tqdm.auto import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Optional import of torch. If not available, mocks below will run.
try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

console = Console()


# -------------------------
# Utility formatting helpers
# -------------------------
def fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -------------------------
# Core training components
# -------------------------
def train_one_epoch(
    model: Any,
    dataloader: Any,
    loss_fn: Any,
    optimizer: Any,
    device: Any,
    epoch_idx: int,
    total_epochs: int,
) -> Dict[str, float]:
    """
    Train the model for a single epoch showing live per-batch tqdm with ETA and dynamic loss.

    Parameters:
        model: A model callable such that outputs = model(inputs).
        dataloader: Iterable yielding (inputs, labels). Can be PyTorch Dataloader.
        loss_fn: Loss function compatible with model outputs.
        optimizer: Optimizer with .zero_grad() and .step() (if torch) or a mock.
        device: Device to move tensors to (if torch). If not torch, ignored.
        epoch_idx: current epoch index (1-based).
        total_epochs: number of total epochs.

    Returns:
        A dict: {"avg_loss": float, "avg_acc": float}
    """
    model.train() if hasattr(model, "train") else None
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # use tqdm to show per-batch progress; leave=False to avoid clutter between epochs
    loop = tqdm(
        enumerate(dataloader, 1),
        total=len(dataloader) if hasattr(dataloader, "__len__") else None,
        desc=f"Epoch {epoch_idx}/{total_epochs} [train]",
        ncols=120,
    )

    epoch_start = time.time()
    for batch_idx, (X, y) in loop:
        # If torch is available and inputs are tensors, move to device
        if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            X = X.to(device)
            y = y.to(device)

        # Forward
        outputs = model(X)

        # Compute loss
        loss = loss_fn(outputs, y)

        # Backward & step (if torch-style optimizer)
        if TORCH_AVAILABLE and isinstance(loss, torch.Tensor):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = float(loss.detach().cpu().item())
        else:
            # If using mock loss (float), just call optimizer.step() if available
            batch_loss = float(loss)
            if hasattr(optimizer, "step"):
                optimizer.step()

        # Metrics (assumes classification; adapt if regression)
        # Try to compute accuracy if outputs are logits/tensor-like
        batch_size = len(y)
        total_samples += batch_size
        correct = 0
        try:
            if TORCH_AVAILABLE and isinstance(outputs, torch.Tensor):
                preds = outputs.argmax(dim=1)
                correct = int((preds == y).sum().item())
            else:
                # outputs as (batch, num_classes) list-like: choose argmax via python
                preds = [int(max(enumerate(o), key=lambda t: t[1])[0]) for o in outputs]
                correct = sum(int(p == int(true)) for p, true in zip(preds, y))
        except Exception:
            correct = 0

        total_correct += correct
        total_loss += batch_loss * batch_size

        # ETA info from tqdm, plus our own postfix (Loss)
        elapsed = time.time() - epoch_start
        avg_loss_so_far = total_loss / total_samples
        acc_so_far = total_correct / total_samples if total_samples > 0 else 0.0

        loop.set_postfix(
            {
                "Loss": f"{avg_loss_so_far:.4f}",
                "BatchLoss": f"{batch_loss:.4f}",
                "Acc": f"{acc_so_far:.4f}",
                "Elapsed": fmt_time(elapsed),
            }
        )

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0

    return {"avg_loss": avg_loss, "avg_acc": avg_acc}


def validate_one_epoch(
    model: Any,
    dataloader: Any,
    loss_fn: Any,
    device: Any,
    epoch_idx: int,
    total_epochs: int,
) -> Dict[str, float]:
    """
    Validation loop with per-batch progress.
    """
    model.eval() if hasattr(model, "eval") else None
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    loop = tqdm(
        enumerate(dataloader, 1),
        total=len(dataloader) if hasattr(dataloader, "__len__") else None,
        desc=f"Epoch {epoch_idx}/{total_epochs} [val]  ",
        ncols=120,
        colour="blue",
    )

    val_start = time.time()
    # If torch: use inference mode / no_grad
    torch_context = (
        torch.inference_mode if TORCH_AVAILABLE else (lambda: DummyContext())
    )

    with torch_context():
        for batch_idx, (X, y) in loop:
            if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
                X = X.to(device)
                y = y.to(device)

            outputs = model(X)
            loss = loss_fn(outputs, y)
            batch_loss = (
                float(loss.detach().cpu().item())
                if (TORCH_AVAILABLE and isinstance(loss, torch.Tensor))
                else float(loss)
            )

            batch_size = len(y)
            total_samples += batch_size

            # accuracy
            try:
                if TORCH_AVAILABLE and isinstance(outputs, torch.Tensor):
                    preds = outputs.argmax(dim=1)
                    correct = int((preds == y).sum().item())
                else:
                    preds = [
                        int(max(enumerate(o), key=lambda t: t[1])[0]) for o in outputs
                    ]
                    correct = sum(int(p == int(true)) for p, true in zip(preds, y))
            except Exception:
                correct = 0

            total_correct += correct
            total_loss += batch_loss * batch_size

            elapsed = time.time() - val_start
            avg_loss_so_far = total_loss / total_samples
            acc_so_far = total_correct / total_samples if total_samples > 0 else 0.0

            loop.set_postfix(
                {
                    "Loss": f"{avg_loss_so_far:.4f}",
                    "BatchLoss": f"{batch_loss:.4f}",
                    "Acc": f"{acc_so_far:.4f}",
                    "Elapsed": fmt_time(elapsed),
                }
            )

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0

    return {"avg_loss": avg_loss, "avg_acc": avg_acc}


# -------------------------
# Main train function
# -------------------------
def train(
    model: Any,
    train_dataloader: Any,
    test_dataloader: Any,
    optimizer: Any,
    loss_fn: Any,
    epochs: int,
    device: Any,
) -> Dict[str, list]:
    """
    Full training + validation loop with rich output and timing.

    Returns results dict with lists for train_loss, train_acc, val_loss, val_acc.
    """
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Fancy header
    header = Panel.fit(
        f"[bold white on dark_green] STARTING TRAINING [/]\n\n"
        f"[bold]Start Time:[/] {now_str()}\n"
        f"[bold]Device:[/] {device}\n"
        f"[bold]Epochs:[/] {epochs}",
        title="🚀 Training Engine",
        subtitle="engine.py",
        padding=(1, 2),
    )
    console.print(header)

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        elapsed_since_start = time.time() - total_start
        console.print(
            f"[yellow]Epoch {epoch}/{epochs} — elapsed: [bold]{fmt_time(elapsed_since_start)}[/bold][/yellow]"
        )

        # Train one epoch
        t_res = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=epochs,
        )

        # Validate one epoch
        v_res = validate_one_epoch(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            epoch_idx=epoch,
            total_epochs=epochs,
        )

        # Epoch summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Phase", style="dim", width=12)
        table.add_column("Loss", justify="right")
        table.add_column("Accuracy", justify="right")

        table.add_row("Train", f"{t_res['avg_loss']:.4f}", f"{t_res['avg_acc']:.4f}")
        table.add_row("Val", f"{v_res['avg_loss']:.4f}", f"{v_res['avg_acc']:.4f}")

        console.print(table)

        # Save results
        results["train_loss"].append(t_res["avg_loss"])
        results["train_acc"].append(t_res["avg_acc"])
        results["val_loss"].append(v_res["avg_loss"])
        results["val_acc"].append(v_res["avg_acc"])

    total_time = time.time() - total_start
    footer = Panel.fit(
        f"[bold white on dark_blue] TRAINING COMPLETE 🎉 [/]\n\n"
        f"[bold]Finish Time:[/] {now_str()}\n"
        f"[bold]Total Duration:[/] {fmt_time(total_time)}",
        title="✅ Done",
        padding=(1, 2),
    )
    console.print(footer)

    return results


# -------------------------
# Small helpers & mocks for demo
# -------------------------
class DummyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class MockModel:
    """A tiny mock model that works with or without PyTorch installed."""

    def __init__(self, num_classes=3, use_torch=TORCH_AVAILABLE):
        self.num_classes = num_classes
        self.use_torch = use_torch
        if use_torch:
            self._w = torch.randn((num_classes, 10), dtype=torch.float32) * 0.01
        else:
            import random

            self._seed = random.Random(42)

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, X):
        # If torch and X is a tensor-like -> return logits tensor
        if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            # fake linear: reduce spatial dims to 10 if needed, then matmul
            flat = X.view(X.size(0), -1)[:, :10].float()
            logits = flat @ self._w.T
            return logits
        else:
            # produce python-native logits: list of lists
            batch_size = len(X)
            out = []
            for _ in range(batch_size):
                # make pseudo-random-ish scores
                scores = [self._seed.random() for _ in range(self.num_classes)]
                out.append(scores)
            return out


class MockOptimizer:
    def zero_grad(self):
        pass

    def step(self):
        pass


class MockLoss:
    """Return a scalar loss. If torch is available, return a tensor loss."""

    def __call__(self, outputs, labels):
        if TORCH_AVAILABLE and isinstance(outputs, torch.Tensor):
            # use simple cross-entropy-like surrogate: negative max logit of true class
            # labels expected as tensor of ints
            batch_size = outputs.size(0)
            # pick predicted scores at label indices safely
            idx = torch.arange(batch_size, dtype=torch.long)
            chosen = outputs[idx, labels]
            loss = -chosen.mean()
            return loss
        else:
            # outputs is list of lists, labels list -> compute naive loss
            total = 0.0
            for o, l in zip(outputs, labels):
                total += -o[int(l)]
            return total / len(outputs)


class MockDataset:
    """Simple iterable that returns (inputs, labels). Works with or without torch."""

    def __init__(
        self,
        n_samples=200,
        input_shape=(3, 32, 32),
        n_classes=3,
        use_torch=TORCH_AVAILABLE,
    ):
        self.n = n_samples
        self.shape = input_shape
        self.n_classes = n_classes
        self.use_torch = use_torch

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            if self.use_torch:
                x = torch.randn(self.shape)
                y = torch.tensor(i % self.n_classes, dtype=torch.long)
                yield x, y
            else:
                # python-native arrays
                x = [[[0.0]]]  # placeholder; MockModel ignores contents
                y = i % self.n_classes
                yield x, y


class MockDataLoader:
    """Simple dataloader that groups samples into batches."""

    def __init__(self, dataset, batch_size=16):
        self.dataset = dataset
        self.batch_size = batch_size
        self._len = (len(dataset) + batch_size - 1) // batch_size

    def __len__(self):
        return self._len

    def __iter__(self):
        batch_x, batch_y = [], []
        for i, (x, y) in enumerate(iter(self.dataset), 1):
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) == self.batch_size:
                if TORCH_AVAILABLE:
                    yield torch.stack(batch_x), torch.stack(batch_y)
                else:
                    yield batch_x, batch_y
                batch_x, batch_y = [], []
        if batch_x:
            if TORCH_AVAILABLE:
                yield torch.stack(batch_x), torch.stack(batch_y)
            else:
                yield batch_x, batch_y


# -------------------------
# Demo runner (executes when file run directly)
# -------------------------
if __name__ == "__main__":
    console.print(
        Panel(
            "[bold magenta]Demo run of engine.py[/bold magenta]\nThis creates mock model/data and runs 2 epochs to show logging."
        ),
        justify="center",
    )

    # Decide device
    device = "cpu"
    if TORCH_AVAILABLE:
        device = torch.device("cpu")

    # Create mocks (these will be replaced by your real objects in normal usage)
    mock_model = MockModel(num_classes=5)
    mock_train_ds = MockDataset(n_samples=400, input_shape=(3, 32, 32), n_classes=5)
    mock_val_ds = MockDataset(n_samples=100, input_shape=(3, 32, 32), n_classes=5)
    mock_train_loader = MockDataLoader(mock_train_ds, batch_size=32)
    mock_val_loader = MockDataLoader(mock_val_ds, batch_size=32)
    mock_optimizer = MockOptimizer()
    mock_loss = MockLoss()

    results = train(
        model=mock_model,
        train_dataloader=mock_train_loader,
        test_dataloader=mock_val_loader,
        optimizer=mock_optimizer,
        loss_fn=mock_loss,
        epochs=2,
        device=device,
    )

    console.print("\n[bold green]Demo complete.[/bold green]")
    # Print final small summary
    summary_table = Table(title="Demo Results", header_style="bold cyan")
    summary_table.add_column("Epoch", justify="right")
    summary_table.add_column("Train Loss", justify="right")
    summary_table.add_column("Train Acc", justify="right")
    summary_table.add_column("Val Loss", justify="right")
    summary_table.add_column("Val Acc", justify="right")
    for i in range(len(results["train_loss"])):
        summary_table.add_row(
            str(i + 1),
            f"{results['train_loss'][i]:.4f}",
            f"{results['train_acc'][i]:.4f}",
            f"{results['val_loss'][i]:.4f}",
            f"{results['val_acc'][i]:.4f}",
        )
    console.print(summary_table)