import os
import glob

__all__ = ["get_directories", "get_extension", "get_files_with_extension", "add_extension", "remove_extension"]

def get_extension(path: str) -> str:
    """
    This function returns the extension of a file.
    Parameters
    ----------
    path : str

    Returns
    -------
    extension : str
        Extension of the file.
    """
    _, extension = os.path.splitext(path)
    extension = extension.replace(".", "")
    return extension


def remove_extension(path: str) -> str:
    """
    This function removes the extension of a file.
    Parameters
    ----------
    path : str

    Returns
    -------
    filename : str
        Filename without extension.
    """
    filename, _ = os.path.splitext(path)
    return filename


def add_extension(path: str, extension: str) -> str:
    """
    This function adds an extension to a file.
    Parameters
    ----------
    path : str
    extension : str

    Returns
    -------
    path : str
        Path with the extension.
    """
    if get_extension(path) != extension:
        path = f"{path}.{extension}"
    return path


def get_files_with_extension( path: str, extension: str ) -> list:
    """
    This function returns all the path to the files inside a folder with a specific extension.

    Parameters:
        path: string
            Path of the folder where to search for files.
        extension: string
            Extension to search for.

    Returns:

    List of all the files inside the specified folder with the desired extension.

    """
    return glob.glob( os.path.join( path, f"*.{extension}" ) )


def get_directories( path: str ) -> list:
    """
    This function returns all the path to the files inside a folder with a specific extension.

    Parameters:
    path: string
        Path of the folder where to search for files.

    Returns:

    List of all the files inside the specified folder with the desired extension.

    """
    directories = [os.path.join( path, subpath ) for subpath in os.listdir(path) if os.path.isdir(os.path.join( path, subpath ))]
    return directories


if __name__ == "__main__":
    pass
