import os


def save_errors(images, y_true, y_pred):
    os.makedirs("research/artifacts/errors/fp", exist_ok=True)
    os.makedirs("research/artifacts/errors/fn", exist_ok=True)

    for i, (img, t, p) in enumerate(zip(images, y_true, y_pred)):
        if t == 0 and p == 1:
            img.save(f"research/artifacts/errors/fp/{i}.png")
        elif t == 1 and p == 0:
            img.save(f"research/artifacts/errors/fn/{i}.png")
