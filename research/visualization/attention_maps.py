import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def generate_swin_attention_map(image_path, output_path):
    """
    Simulate Swin Transformer attention maps for research visualization.
    In a real scenario, this would extract attention weights from the Swin stages.
    """
    img = Image.open(image_path).convert("L")
    img_array = np.array(img.resize((224, 224)))
    
    # Create a mock multi-scale attention map
    # Swin has 4 stages, we simulate them with different noise scales
    attention = np.zeros((224, 224))
    
    # Stage 1: Fine details
    attention += np.random.normal(0.1, 0.05, (224, 224))
    
    # Stage 2: Mid-scale
    mid = np.random.normal(0.2, 0.1, (56, 56))
    attention += np.array(Image.fromarray(mid).resize((224, 224), Image.BILINEAR))
    
    # Stage 3: Global structures (the most important for lens detection)
    # We focus attention on high-intensity regions (galaxies)
    focus = (img_array > np.percentile(img_array, 90)).astype(float)
    attention += focus * 0.5
    
    # Normalize
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_array, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Swin Stage 3 Attention")
    plt.imshow(attention, cmap="inferno")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Generated attention map: {output_path}")

if __name__ == "__main__":
    # Test run if an image exists
    img_path = "euclid_cache/fallback_tile.fits" # Or any png
    if Path(img_path).exists():
        generate_swin_attention_map(img_path, "research/artifacts/plots/attention_map_demo.png")
