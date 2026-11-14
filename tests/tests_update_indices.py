import numpy as np
import pytest
import torch

from alcv.utils.update_indices import update_labeled_and_unlabeled_indices


@pytest.mark.parametrize("kind", ["python", "numpy", "torch"])
def test_update_accepts_integer_index_types(kind):
    convert = {"python": list, "numpy": lambda x: np.array(x, dtype=np.int64), "torch": lambda x: torch.tensor(x, dtype=torch.long)}[kind]
    labeled, unlabeled = convert([2, 5]), convert([11, 23, 37, 41])
    new_labeled, new_unlabeled = update_labeled_and_unlabeled_indices(labeled, unlabeled, convert([23, 41]))

    assert new_labeled == [2, 5, 23, 41]
    assert new_unlabeled == [11, 37]
    assert all(type(i) is int for i in new_labeled + new_unlabeled)
    assert not set(new_labeled) & set(new_unlabeled)
    assert list(labeled) == [2, 5]
    assert list(unlabeled) == [11, 23, 37, 41]


def test_update_accepts_mixed_integer_types():
    labeled = [np.int64(2), torch.tensor(5)]
    unlabeled = [torch.tensor(11), np.int64(23), 37]
    new_labeled, new_unlabeled = update_labeled_and_unlabeled_indices(labeled, unlabeled, torch.tensor([23, 37]))

    assert new_labeled == [2, 5, 23, 37]
    assert new_unlabeled == [11]
    assert all(type(i) is int for i in new_labeled + new_unlabeled)


@pytest.mark.parametrize("labeled,unlabeled,selected", [
    ([2, 5], [11, 23], [23, 23]),
    ([2, 5], [11, 23], [999]),
    ([2, 5], [11, 23], [2]),
    ([2, 2], [11, 23], [11]),
    ([2, 5], [11, 11], [11]),
    ([2, 5], [5, 11], [11]),
])
def test_update_rejects_invalid_pool_membership(labeled, unlabeled, selected):
    with pytest.raises(ValueError):
        update_labeled_and_unlabeled_indices(labeled, unlabeled, selected)


def test_update_preserves_invariants_over_multiple_rounds():
    labeled, unlabeled = [0, 1], list(range(2, 10))

    for selected in ([2, 3], np.array([4, 5]), torch.tensor([6, 7]), [8, 9]):
        previous_size = len(labeled)
        labeled, unlabeled = update_labeled_and_unlabeled_indices(labeled, unlabeled, selected)

        assert len(labeled) == previous_size + 2
        assert len(labeled) + len(unlabeled) == 10
        assert len(labeled) == len(set(labeled))
        assert len(unlabeled) == len(set(unlabeled))
        assert not set(labeled) & set(unlabeled)
        assert set(labeled) | set(unlabeled) == set(range(10))

    assert unlabeled == []


def test_update_accepts_empty_selection():
    labeled, unlabeled = update_labeled_and_unlabeled_indices([2, 5], [11, 23], [])
    assert labeled == [2, 5]
    assert unlabeled == [11, 23]
