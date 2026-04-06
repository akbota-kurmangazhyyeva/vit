import argparse
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import timm


def main(output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "levit_192.fb_dist_in1k"
    model = timm.create_model(model_name, pretrained=True, num_classes=10)

    state = torch.load(
        "levit_192.fb_dist_in1k_cifar10_final.pth",
        map_location="cpu"
    )
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_dataset = datasets.CIFAR10(
        root="/home/esrg/Desktop/efficient_vit_new/data",
        train=False,
        download=True,
        transform=preprocess
    )

    indices = list(range(1000))
    test_subset = Subset(test_dataset, indices)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    predictions = {}
    true_labels = {}

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            _, predicted_idx = torch.max(output, 1)

            start_idx = batch_idx * test_loader.batch_size
            for j in range(inputs.size(0)):
                idx = start_idx + j
                predictions[idx] = predicted_idx[j].item()
                true_labels[idx] = labels[j].item()

    correct = sum(
        1 for idx in predictions
        if predictions[idx] == true_labels[idx]
    )
    accuracy = correct / float(len(predictions))
    print("levit_192: Accuracy on {} test images: {:.4f}".format(len(predictions), accuracy))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, "output_levit_192.csv")
    with open(output_file, "w") as f:
        f.write("index,prediction,true_label\n")
        for idx in sorted(predictions.keys()):
            f.write("{},{},{}\n".format(idx, predictions[idx], true_labels[idx]))

    print("levit_192: Saved predictions for {} images to {}".format(len(predictions), output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on CIFAR-10 test subset")
    parser.add_argument("output_dir", type=str, help="Output directory for results")
    args = parser.parse_args()
    main(args.output_dir)