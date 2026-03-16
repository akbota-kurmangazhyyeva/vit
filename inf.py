import argparse
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import timm
import random

def main(output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_model_name = "levit_conv_192.fb_dist_in1k"
    model = timm.create_model(full_model_name, pretrained=True, num_classes=10)
    model.load_state_dict(torch.load('levit_conv_192.fb_dist_in1k_cifar10_final.pth', map_location=torch.device('cpu'), weights_only=True))
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_dataset = datasets.CIFAR10(root='/home/esrg/Desktop/efficient_vit_new/data', train=False, download=True, transform=preprocess)

    indices = list(range(1000))
    test_subset = Subset(test_dataset, indices)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    predictions = {}
    true_labels = {}
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            _, predicted_idx = torch.max(output, 1)
            predictions[i] = predicted_idx.item()
            true_labels[i] = labels.item()
            print(f"Image {i}: Predicted class index: {predicted_idx.item()}")

    correct = sum(1 for idx in predictions if predictions[idx] == true_labels[idx])
    accuracy = correct / len(predictions)
    print(f"levit_conv_192: Accuracy on 1000 test images: {accuracy:.4f}")

    output_file = os.path.join(output_dir, "output_levit_conv_192.csv")
    with open(output_file, "w") as f:
        f.write("index,prediction,true_label\n")
        for idx in predictions:
            f.write(f"{idx},{predictions[idx]},{true_labels[idx]}\n")
    print(f"levit_conv_192: Saved predictions for {len(predictions)} images to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on CIFAR-10 test subset")
    parser.add_argument('output_dir', type=str, help='Output directory for results')
    args = parser.parse_args()
    main(args.output_dir)