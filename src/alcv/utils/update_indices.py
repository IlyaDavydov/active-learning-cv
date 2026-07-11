def update_labeled_and_unlabeled_indices(
    labeled_indices,
    unlabeled_indices,
    selected_indices
):
    labeled_indices = list(labeled_indices)
    unlabeled_indices = list(unlabeled_indices)
    selected_indices = list(selected_indices)

    selected_indices_set = set(selected_indices)

    new_labeled_indices = labeled_indices + selected_indices

    new_unlabeled_indices = [
        index
        for index in unlabeled_indices
        if index not in selected_indices_set
    ]

    return new_labeled_indices, new_unlabeled_indices