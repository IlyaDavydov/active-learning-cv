from operator import index as integer_index


def update_labeled_and_unlabeled_indices(labeled_indices, unlabeled_indices, selected_indices,):
    labeled = [integer_index(i) for i in labeled_indices]
    unlabeled = [integer_index(i) for i in unlabeled_indices]
    selected = [integer_index(i) for i in selected_indices]

    labeled_set = set(labeled)
    unlabeled_set = set(unlabeled)
    selected_set = set(selected)

    if len(labeled_set) != len(labeled):
        raise ValueError("Labeled pool contains duplicate indices")

    if len(unlabeled_set) != len(unlabeled):
        raise ValueError("Unlabeled pool contains duplicate indices")

    if labeled_set & unlabeled_set:
        raise ValueError("Labeled and unlabeled pools overlap")

    if len(selected_set) != len(selected):
        raise ValueError("Selected indices contain duplicates")

    if not selected_set <= unlabeled_set:
        raise ValueError(
            "Selected indices must belong to the unlabeled pool"
        )

    new_labeled = labeled + selected
    new_unlabeled = [
        i for i in unlabeled if i not in selected_set
    ]

    return new_labeled, new_unlabeled