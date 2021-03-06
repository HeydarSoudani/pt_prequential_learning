
import random
from typing import List, Tuple
import torch
from torch.utils.data import Sampler, Dataset


class PtSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label = {}
        assert hasattr(
            dataset, "labels"
        ), "TaskSampler needs a dataset with a field 'label' containing the labels of all images."
        for item, label in enumerate(dataset.labels):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        # random.choices(
                        #     self.items_per_label[label], k=self.n_shot + self.n_query
                        # )
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    # pylint: enable=not-callable
                    for label in random.sample(self.items_per_label.keys(), self.n_way)
                ]
            )

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
        """

        input_data.sort(key = lambda input_data: input_data[1])

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
          (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        
        all_labels = torch.tensor(
          [x[1] for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))

        support_images, query_images = torch.split(all_images, [self.n_shot, self.n_query], dim=1)
        support_labels, query_labels = torch.split(all_labels, [self.n_shot, self.n_query], dim=1)
        
        # sel_classes = np.random.choice(self.n_way, self.n_query_way, replace=False)
        # query_images = query_images[sel_classes]
        # query_labels = query_labels[sel_classes]

        return (
          support_images,
          support_labels,
          query_images,
          query_labels
        )

