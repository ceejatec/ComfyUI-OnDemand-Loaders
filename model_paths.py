"""
Model names and the files they are stored in.

A model named "zimage/characters/jane_doe" is stored as <models folder>/zimage/characters/jane_doe<extension>,
with the extension taken from the downloaded file unless the name already ends in a model extension.
This module has no ComfyUI imports, so it can be tested on its own.
"""
import os
import re
from urllib.parse import urlparse, unquote

# Characters Windows does not allow in file names, plus backslash (names use / for folders) and control characters
_INVALID_CHARS = re.compile(r'[<>:"|?*\\\x00-\x1f]')
_RESERVED_WINDOWS_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def check_model_name(name):
    """
    Returns why name cannot be used as a model file path within a models folder, or None if it can.
    """
    if not name or name != name.strip():
        return "it is empty or starts or ends with a space"
    if name == "None":
        return "'None' is reserved for no model"
    if "\\" in name:
        return "use / to separate folders"
    for part in name.split("/"):
        if not part:
            return "it starts or ends with /, or contains //"
        if part in (".", ".."):
            return "'.' and '..' cannot be used as folder names"
        invalid = sorted(set(_INVALID_CHARS.findall(part)))
        if invalid:
            return "it contains characters that cannot be used in file names: " + " ".join(repr(c) for c in invalid)
        if part[-1] in ". ":
            return "a folder or file name ends with a dot or space"
        if part.split(".")[0].strip().upper() in _RESERVED_WINDOWS_NAMES:
            return f"'{part}' is a reserved file name on Windows"
    return None


def check_server_filename(filename):
    """
    Returns why a filename sent by a server cannot be used, or None if it can.
    """
    if "/" in filename:
        return "it contains /"
    return check_model_name(filename)


def model_extension(name, extensions):
    """
    Returns the model extension (e.g. ".safetensors") that name ends with, or "" if it does not end with one.
    """
    extension = os.path.splitext(name)[1].lower()
    return extension if extension in extensions else ""


def named_model_path(models_dir, name, extension, extensions):
    """
    Returns the path of the file for a model name: the name plus extension, unless it already ends in a model extension.
    The name must pass check_model_name; the path is also checked to be inside models_dir.
    """
    filename = name if model_extension(name, extensions) else name + extension
    path = os.path.normpath(os.path.join(models_dir, *filename.split("/")))
    if not is_within(models_dir, path):
        raise ValueError(f"'{name}' does not resolve to a path inside {models_dir}")
    return path


def find_named_model_file(models_dir, name, extensions):
    """
    Returns the existing file for a model name (the name itself if it ends in a model extension,
    else the name with any model extension), or None.
    """
    if model_extension(name, extensions):
        candidates = [name]
    else:
        candidates = [name + extension for extension in sorted(extensions)]
    for candidate in candidates:
        path = os.path.normpath(os.path.join(models_dir, *candidate.split("/")))
        if is_within(models_dir, path) and os.path.isfile(path):
            return path
    return None


def is_within(base_dir, path):
    """
    True if path is base_dir or inside it (compared lexically, so folders linked from elsewhere still count).
    """
    base_dir = os.path.normcase(os.path.normpath(os.path.abspath(base_dir)))
    path = os.path.normcase(os.path.normpath(os.path.abspath(path)))
    return os.path.commonpath([base_dir, path]) == base_dir


def relative_model_path(models_dir, path):
    """
    The path of a model file relative to its models folder, with / separators, as ComfyUI's loaders take it.
    """
    return os.path.relpath(path, models_dir).replace(os.sep, "/")


def server_filename(model_url, response, extensions):
    """
    Returns the filename a download response provides: from its Content-Disposition header, else the last
    part of the URL path if it ends in a model extension. Returns None if there is neither.
    """
    content_disposition = response.headers.get("Content-Disposition", "")
    # RFC 6266: filename*=UTF-8''<percent-encoded> takes precedence over filename=
    match = re.search(r"filename\*\s*=\s*[\w-]+'[^']*'([^;]+)", content_disposition, re.IGNORECASE)
    if match:
        return unquote(match.group(1).strip().strip('"')) or None
    match = re.search(r'filename\s*=\s*(?:"([^"]*)"|([^;]+))', content_disposition, re.IGNORECASE)
    if match:
        filename = (match.group(1) if match.group(1) is not None else match.group(2)).strip()
        if filename:
            return filename

    for url in (model_url, response.url):
        last_part = unquote(urlparse(url).path.rsplit("/", 1)[-1])
        if model_extension(last_part, extensions):
            return last_part
    return None
