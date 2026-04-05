import argparse
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


def main(output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "WinKawaks/vit-small-patch16-224"

    # Older Transformers versions for Python 3.6 use AutoFeatureExtractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=10,
        ignore_mismatched_sizes=True,
        use_safetensors=False  # safer for torch 1.10
    )

    state = torch.load(
        "vit-small-patch16-224_cifar10_final.pth",
        map_location="cpu"
    )
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Keep ToTensor, then convert each image back to PIL before the feature extractor
    test_dataset = datasets.CIFAR10(
        root="/home/esrg/Desktop/efficient_vit_new/data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    indices = list(range(1000))
    test_loader = DataLoader(
        Subset(test_dataset, indices),
        batch_size=64,
        shuffle=False
    )

    to_pil = transforms.ToPILImage()

    correct = 0
    n = 0
    preds_log = []
    labels_log = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            pil_images = [to_pil(img) for img in imgs]

            batch = feature_extractor(
                images=pil_images,
                return_tensors="pt"
            )

            # Older BatchFeature handling: move tensors manually
            batch = dict((k, v.to(device)) for k, v in batch.items())
            labels = labels.to(device)

            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()
            n += labels.size(0)

            preds_log.extend(preds.cpu().tolist())
            labels_log.extend(labels.cpu().tolist())

    acc = float(correct) / float(n)
    print("vit_s: Accuracy on {} test images: {:.4f}".format(n, acc))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "output_vit_s.csv")
    with open(output_path, "w") as f:
        f.write("index,prediction,true_label\n")
        for i, (p, t) in enumerate(zip(preds_log, labels_log)):
            f.write("{},{},{}\n".format(i, p, t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    main(args.output_dir)