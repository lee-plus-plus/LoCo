import os
import re
import pickle
import inspect
from typing import Callable
from functools import wraps


def get_args_dict(func, *args, **kwargs):
    """Get a dictionary of bound arguments for a function.

    Args:
        func: The function to inspect.
        *args: Positional arguments passed to the function.
        **kwargs: Keyword arguments passed to the function.

    Returns:
        dict: A dictionary of bound arguments.
    """
    # Get the function signature
    sig = inspect.signature(func)
    # Bind the arguments to the signature
    bound_args = sig.bind(*args, **kwargs)
    # Apply default values if any
    bound_args.apply_defaults()
    # Return the dictionary of arguments
    return bound_args.arguments


def validate_filename(filename, replace_to=""):
    """Validate and sanitize a filename by removing illegal characters.

    Args:
        filename (str): The filename to validate.
        replace_to (str): The string to replace illegal characters with. Defaults to "".

    Returns:
        str: A sanitized filename.
    """
    # Define a regex pattern for illegal characters in filenames
    illegal_str = r"[\/\\\:\*\?\"\<\>\|]"  # '/\:*?"<>|'
    # Replace illegal characters with the specified string
    return re.sub(illegal_str, replace_to, filename)


def default_format(args_dict: dict) -> str:
    """Default formatting function for cache filenames."""
    return ",".join(f"{key}={repr(value)}" for key, value in args_dict.items())


def file_cache(cache_dir: str = "cache", format: str | Callable[[dict], str] = default_format, enabled: bool = True):
    """Decorator to cache the results of a function to disk."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(cache_dir, exist_ok=True)

            # Get the dictionary of bound arguments
            args_dict = get_args_dict(func, *args, **kwargs)

            filename = format.format(**args_dict) if isinstance(format, str) else format(args_dict)
            filename = validate_filename(filename)
            cache_filename = os.path.join(cache_dir, filename)

            # If the cache file exists, load and return the cached result
            if os.path.exists(cache_filename):
                with open(cache_filename, "rb") as f:
                    return pickle.load(f)

            # Otherwise, compute the result and cache it
            result = func(*args, **kwargs)
            with open(cache_filename, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper if enabled else func

    return decorator
