import time


def benchmark(model, loader):
    start = time.time()

    for batch in loader:
        model(batch)

    total = time.time() - start
    print(f"Total time: {total:.2f}s")
