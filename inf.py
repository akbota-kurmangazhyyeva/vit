import argparse, os, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import ViTFeatureExtractor, ViTForImageClassification


def main(output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AutoImageProcessor did not exist in 4.18; use ViTFeatureExtractor directly.
    processor = ViTFeatureExtractor.from_pretrained(
        "WinKawaks/vit-small-patch16-224"
    )

    # ignore_mismatched_sizes was added after 4.18, so we swap the head manually.
    model = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224"
    )
    # Replace the classifier head to match the target number of labels (10).
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, 10)

    state = torch.load(
        "vit-small-patch16-224_cifar10_final.pth", map_location="cpu"
    )
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    test_dataset = datasets.CIFAR10(
        root="/home/esrg/Desktop/efficient_vit_new/data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    indices = list(range(1000))
    test_loader = DataLoader(
        Subset(test_dataset, indices), batch_size=64, shuffle=False
    )

    correct = n = 0
    preds_log, labels_log = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            # ViTFeatureExtractor expects a list of PIL images or numpy arrays.
            # Convert the float32 [0,1] tensor batch → list of uint8 numpy HWC arrays
            # so the feature extractor can normalise them the same way the model
            # was originally trained (mean/std from the pretrained config).
            imgs_np = (imgs.permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
            imgs_list = list(imgs_np)  # list of H×W×C uint8 arrays

            # In 4.18 the feature extractor handles rescaling & normalisation
            # internally, so do NOT pass do_rescale / data_format.
            batch = processor(images=imgs_list, return_tensors="pt")

            # Move every tensor in the batch to the target device manually
            # (.to(device) on the BatchEncoding object was not reliable in 4.18).
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch).logits
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