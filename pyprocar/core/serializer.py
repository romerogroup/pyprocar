from pathlib import Path

import dill


class PickleSerializer:
    """Serializer for Electronic Band Structure using pickle format."""

    def save(self, obj: object, path: Path):
        """Save the EBS to a pickle file.

        Args:
            ebs: The ElectronicBandStructure object to save
            path: Path where to save the pickle file
        """
        with open(path, "wb") as file:
            dill.dump(obj, file)

    def load(self, path: Path):
        """Load an EBS from a pickle file.

        Args:
            path: Path to the pickle file

        Returns:
            The loaded ElectronicBandStructure object
        """
        with open(path, "rb") as file:
            return dill.load(file)


_PICKLE = PickleSerializer()
SERIALIZERS = {"pickle": _PICKLE, "pkl": _PICKLE}


def get_serializer(path: Path | str):
    """Get the serializer for the given path."""
    if isinstance(path, str):
        path = Path(path)
    return SERIALIZERS[path.suffix[1:]]
