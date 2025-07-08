"""File utilities for CLISOPS."""

import os
import pathlib


class FileMapper:
    """
    Class to represent a set of files that exist in the same directory as one object.

    Parameters
    ----------
    file_list : list of str
        The list of files to represent. If dirpath is not provided, these should be full file paths.
    dirpath : str or Path, optional
        The directory path where the files exist. Default is None.
        If dirpath is not provided, it will be deduced from the file paths provided in file_list.

    Attributes
    ----------
    file_list : list of str
        List of file names of the files represented.
    file_paths : list of str or list of Path
        List of full file paths of the files represented.
    dirpath : str or Path
        The directory path where the files exist. Default is None.
        If dirpath is not provided, it will be deduced from the file paths provided in file_list.
    """

    def __init__(self, file_list: list[str], dirpath: str | pathlib.Path | None = None):
        """
        Initialize the FileMapper with a list of files and an optional directory path.

        Parameters
        ----------
        file_list : list of str
            List of file names or full paths to the files.
        dirpath : str or Path, optional
            The directory path where the files exist. Default is None.
            If dirpath is not provided, it will be deduced from the file paths provided in file_list.
        """
        self.dirpath = dirpath
        self.file_list = file_list

        self._resolve()

    def _resolve(self):
        if not self.dirpath:
            first_dir = os.path.dirname(self.file_list[0])
            if os.path.dirname(os.path.commonprefix(self.file_list)) == first_dir:
                self.dirpath = first_dir
            else:
                raise Exception("File inputs are not from the same directory so cannot be resolved.")

        self.file_list = [os.path.basename(fpath) for fpath in self.file_list]
        self.file_paths = [os.path.join(self.dirpath, fname) for fname in self.file_list]
        # check all files exist on filesystem
        if not all(os.path.isfile(file) for file in self.file_paths):
            raise FileNotFoundError("Some files could not be found.")


def is_file_list(coll: list[str]) -> bool:
    """
    Check whether a collection is a list of files.

    Parameters
    ----------
    coll : list
        Collection to check.

    Returns
    -------
    bool
        True if the collection is a list of files, else returns False.
    """
    # check if a collection is a list of files
    if not isinstance(coll, list):
        raise Exception(f"Expected collection as a list, have received {type(coll)}")

    if coll[0].startswith("/") or coll[0].startswith("http"):
        return True

    return False
