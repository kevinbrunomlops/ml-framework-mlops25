from pathlib import Path
import json

from src.config import load_config
from src.utils import set_seed, get_device
from src.dataset import download_cifar10, make_loaders 
from src.model import SimpleCNN
from src.train import fit

def main():
    cfg = load_config("params.yaml")

    set_seed(cfg.seed)
    device = get_device()
    print("Device:", device)
    print("Run:", cfg.run_name)

    download_cifar10(cfg.data_dir)
    train_loader, test_loader = make_loaders(
        cfg.data_dir, 
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        )

    model = SimpleCNN(num_classes=10).to(device)
    history = fit(model, train_loader, test_loader,
                   epochs=cfg.epochs, lr=cfg.lr, device=device)
    
    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))
    print(f"Saved metrics to: {run_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
