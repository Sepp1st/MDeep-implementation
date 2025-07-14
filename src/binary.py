import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from model import NetworkBinary


def train(x_train, y_train, args):
    # 1. Setup device, data loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x_train_cleaned = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_train_final = x_train_cleaned.astype(np.float32)
    x_train_tensor = torch.from_numpy(x_train_final).unsqueeze(1)  # Add channel dimension (1 channel for Conv1D)

    # PyTorch CrossEntropyLoss expects class indices, NOT one-hot vectors.
    y_train_tensor = torch.from_numpy(y_train.argmax(axis=1)).long()

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Initialize model, optimizer, loss
    num_features = x_train.shape[1]  # Number of features in the input data
    model = NetworkBinary(args, num_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    variances = np.var(x_train, axis=0)
    zero_variance_features = np.where(variances == 0)[0]
    if len(zero_variance_features) > 0:
        print(f"WARNING: Found {len(zero_variance_features)} features with zero variance!")
        print("Feature indices:", zero_variance_features)

    # 3. Training loop
    for epoch in range(args.max_epoch):
        model.train()  # Set model to training mode
        total_loss = 0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        if train_accuracy > 99.0:
            print("Early stopping triggered.")
            break

    # 4. Save model
    save_path = f"/kaggle/working/{args.model_dir}/{args.data_dir.split('/')[-1]}/model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved in path: {save_path}")


def eval(x_test, y_test, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    x_test_cleaned = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
    x_test_final = x_test_cleaned.astype(np.float32)
    x_test_tensor = torch.from_numpy(x_test_final).unsqueeze(1)  # Add channel dimension (1 channel for Conv1D)

    y_true_indices = y_test.argmax(axis=1)  # For sklearn metrics

    # Load model
    num_features = x_test.shape[1]  # Number of features in the input data
    model = NetworkBinary(args, num_features).to(device)
    model.load_state_dict(torch.load(f"{args.model_dir}/{args.data_dir.split('/')[-1]}/model.pth"))
    model.eval()  # Set to evaluation mode

    y_pred_probs = []
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(x_test_tensor.to(device))
        # Get probabilities for the positive class
        y_pred_probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

    # Calculate ROC and AUC
    fpr, tpr, _ = metrics.roc_curve(y_true_indices, y_pred_probs)
    auc = metrics.roc_auc_score(y_true_indices, y_pred_probs)
    print(f"AUC Score: {auc}")

    # Plotting (same as before)
    plt.clf()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/{args.result_dir}/result.jpg")
    plt.show()