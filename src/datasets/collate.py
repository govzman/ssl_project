import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    result_batch["images"] = torch.vstack(
        [elem["image"].unsqueeze(0) for elem in dataset_items]
    )
    result_batch["raw_images"] = torch.vstack(
        [elem["raw_image"].unsqueeze(0) for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor(
        [elem["label"] for elem in dataset_items], dtype=torch.long
    )
    return result_batch
