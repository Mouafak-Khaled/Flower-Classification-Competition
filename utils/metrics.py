import torch
import torchmetrics

def accuracy(
    scores: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 17,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Computes the classification accuracy for a multi-class problem.

    This function calculates accuracy by comparing predicted class indices
    (obtained via `argmax`) with ground truth labels. It uses `torchmetrics.Accuracy`
    for robust computation.

    Args:
        scores (torch.Tensor): Model output logits or scores of shape (batch_size, num_classes).
        target (torch.Tensor): Ground truth labels of shape (batch_size,).
        num_classes (int, optional): The total number of classes. Defaults to 17.
        device (str, optional): The device to run the metric computation on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        torch.Tensor: The computed accuracy as a scalar tensor.
    """
    # Convert logits to class indices
    preds = torch.argmax(scores, dim=1)

    # Initialize accuracy metric
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

    return acc_metric(preds, target)