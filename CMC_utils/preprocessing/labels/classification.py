import numpy as np

__all__ = ["label_to_discrete", "discrete_to_label", "label_to_categorical", "categorical_to_label"]

def label_to_discrete( label, classes: tuple ) -> int:
    classes_map = {klass: idx for idx, klass in enumerate(classes)}
    return classes_map[ label ]


def discrete_to_label( label, classes: tuple, **_  ) -> str:
    if isinstance(label, np.ndarray):
        return categorical_to_label(label, classes)

    classes_map = {idx: klass for idx, klass in enumerate(classes)}  # [label]
    return classes_map[label]


def label_to_categorical( label, classes: tuple ) -> np.array:
    discrete_label = label_to_discrete( label, classes )
    return np.eye( len(classes), dtype="int32" )[ discrete_label ].squeeze()


def categorical_to_label( label: np.array, classes: tuple, **_  ) -> str:
    return classes[ np.argmax( label ) ]


if __name__ == "__main__":
    pass
