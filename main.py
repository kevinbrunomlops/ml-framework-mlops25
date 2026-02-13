from src.dataset import download_cifar10, make_loaders

def main():
    download_cifar10("data/raw")
    train_loader, test_loader = make_loaders("data/raw", batch_size=128)
    print("Train batches", len(train_loader))
    print("Test batches", len(test_loader))


if __name__ == "__main__":
    main()
