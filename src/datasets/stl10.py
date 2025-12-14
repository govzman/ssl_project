import json
import logging
import random
from typing import List

import torchvision
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


class STL10(BaseDataset):
    """
    STL10
    """

    def __init__(self, split, *args, **kwargs):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        data = self._get_or_load_data(split)
        super().__init__(data, *args, **kwargs)

    def _get_or_load_data(self, split):
        data_path = ROOT_PATH / "data" / "datasets" / "stl10"
        data = torchvision.datasets.STL10(data_path, split=split, download=True)
        return data
