import argparse, os, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import ViTConfig, ViTForImageClassification

def main(output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ViT-Small/16 expects 224x224, normalized with ImageNet stats
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # CIFAR10 is 32x32, must upscale
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]   # ViT models typically use 0.5/0.5 normalization
        ),
    ])

    model = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True
    )
    state = torch.load('vit-small-patch16-224_cifar10_final.pth', map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    test_dataset = datasets.CIFAR10(
        root='/home/esrg/Desktop/efficient_vit_new/data',
        train=False, download=True,
        transform=transforms.ToTensor(),  # gives [0,1] float tensors
    )
    indices = list(range(1000))
    test_loader = DataLoader(Subset(test_dataset, indices), batch_size=64, shuffle=False)

    correct = n = 0
    preds_log, labels_log = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            # imgs: [B, 3, 32, 32] float in [0,1]
            imgs = preprocess(imgs).to(device)  # -> [B, 3, 224, 224] normalized

            logits = model(pixel_values=imgs).logits
            preds = logits.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            n += labels.size(0)
            preds_log.extend(preds.tolist())
            labels_log.extend(labels.tolist())

    acc = correct / n
    print(f"vit_s: Accuracy on {n} test images: {acc:.4f}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "output_vit_s.csv"), "w") as f:
        f.write("index,prediction,true_label\n")
        for i, (p, t) in enumerate(zip(preds_log, labels_log)):
            f.write(f"{i},{p},{t}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    main(parser.parse_args().output_dir)