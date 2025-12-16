import logging
import random
from typing import List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        data,
        limit=None,
        two_augmentations=False,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            two_augmentations (bool): if True - add second augmented image to batch.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """

        self._data: List[tuple] = data
        self.limit = limit if limit else len(data)
        self.indexes = torch.randint(low=0, high=len(data), size=(self.limit,))
        self.two_augmentations = two_augmentations
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """

        data_tuple = self._data[self.indexes[ind % self.limit]]

        instance_data = {
            "image": data_tuple[0],
            "raw_image": data_tuple[0],
            "label": data_tuple[1],
        }
        if self.two_augmentations:
            instance_data["image2"] = data_tuple[0]
        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._data)

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name in instance_data:
                    instance_data[transform_name] = self.instance_transforms[
                        transform_name
                    ](instance_data[transform_name])
        return instance_data
