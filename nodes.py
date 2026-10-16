import asyncio
import json
import os
import shutil
import requests
import sys
import logging
import threading
from tqdm import tqdm
import folder_paths
import comfy.utils
import comfy.model_management
from pathlib import Path
import importlib.util
import hashlib
import server
from aiohttp import web
from nodes import LoraLoader, UNETLoader, CheckpointLoaderSimple, VAELoader, CLIPLoader,  ControlNetLoader, DualCLIPLoader, CLIPVisionLoader
from comfy_extras.nodes_model_patch import ModelPatchLoader
from .sources import get_source
from .config_transfer import export_sections, plan_import, apply_import, count_changes
from .model_paths import (check_model_name, check_server_filename, check_subfolder, normalize_subfolder, list_subfolders,
                          model_extension, named_model_path, find_named_model_file, relative_model_path, server_filename)


LOG_PREFIX = "[ComfyUI-OnDemand-Loaders]"

# File extensions of model files
MODEL_EXTENSIONS = set(folder_paths.supported_pt_extensions) | {".gguf"}

# Folder in ComfyUI/models that each config.json list downloads to
MODEL_DIRS = {
    "loras": "loras",
    "diffusion_models": "diffusion_models",
    "checkpoints": "checkpoints",
    "vae_models": "vae",
    "clip_models": "text_encoders",
    "clip_vision": "clip_vision",
    "gguf_models": "unet",
    "controlnet_models": "controlnet",
    "model_patches": "model_patches",
}

# ComfyUI folder (folder_paths key) whose files are offered as local models for each config.json list;
# it covers ComfyUI's other folders for that type and extra_model_paths.yaml too
MODEL_FOLDER_KEYS = {
    "loras": "loras",
    "diffusion_models": "diffusion_models",
    "checkpoints": "checkpoints",
    "vae_models": "vae",
    "clip_models": "text_encoders",
    "clip_vision": "clip_vision",
    "gguf_models": "unet_gguf",  # Registered by ComfyUI-GGUF
    "controlnet_models": "controlnet",
    "model_patches": "model_patches",
}

# Drop-down entries for model files that are on disk but not in config.json start with LOCAL_PREFIX,
# and the url field shows LOCAL_URL for them
LOCAL_PREFIX = "[local] "
LOCAL_URL = "<local>"


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.handlers = []
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(f"{LOG_PREFIX} %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(f"Starting dynamic import of nodes.py from ComfyUI-GGUF...")

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
module_path = Path(parent_dir) / 'ComfyUI-GGUF' / '__init__.py'
module_name = "ComfyUI-GGUF"
module_gguf = None
if module_path.exists():
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        module_gguf = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module_gguf
        spec.loader.exec_module(module_gguf)

        logger.info(f"Successfully found and imported UnetLoaderGGUF dynamically.")
    except AttributeError as ae:
        logger.error(f"'UnetLoaderGGUF' not found. Check the class name: {ae}")
        module_gguf = None
    except Exception as e:
        logger.error(f"Error during module execution (nodes.py content error): {e}")
        module_gguf = None

else:
    # Handle the missing file case gracefully
    logger.warning(f"ComfyUI-GGUF installation not found! Expected location: {module_path.parent.as_posix()}")
    logger.warning(f"OnDemand GGUF Loaders will not be available")


def _get_config_path(config_filename="config.json"):
    """
    Returns the path of the configuration file: ONDEMAND_LOADERS_CONFIG_PATH if that file exists, else config.json next to this file.
    """
    config_path_env = os.environ.get('ONDEMAND_LOADERS_CONFIG_PATH')
    if config_path_env and os.path.exists(config_path_env):
        return config_path_env
    return os.path.join(os.path.dirname(__file__), config_filename)


# Function to load configuration
def load_config(config_filename="config.json", quiet=False):

    config_path = _get_config_path(config_filename)
    config_filename = os.path.basename(config_path)

    default_config = {
        "loras": [
            {
                "name": "Lora n1",
                "url": "not_valid_url",
            },
            {
                "name": "Lora n2",
                "url": "not_valid_url"
            }
        ]
    }

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        if not quiet:
            logger.info(f"Successfully loaded configuration from {config_filename}")

        # add None to all lists
        for key in config:
            if isinstance(config[key], list):
                none_entry = {"name": "None", "url": None}
                if none_entry not in config[key]:
                    config[key].insert(0, none_entry)

        return config
    except FileNotFoundError:
        if not quiet:
            logger.warning(f"Configuration file '{config_path}' not found. Using default fallback configuration.")
        return default_config
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from '{config_path}'. Using default fallback configuration.")
        return default_config
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading '{config_path}': {e}. Using default fallback.")
        return default_config


_config_write_lock = threading.Lock()

def _read_config_file(config_path):
    """
    Returns the contents of config.json as stored (without the "None" entries load_config adds), or {} if it does not exist.
    Raises if it cannot be read or parsed; callers must then not overwrite it.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def _write_config_file(config_path, config):
    """
    Replaces config.json with config, atomically so it is never left half-written.
    """
    tmp_path = config_path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(config, f, indent=4)
        f.write("\n")
    os.replace(tmp_path, config_path)

def _update_config_entry(model_type_key, model_name, changes, create=False):
    """
    Applies changes ({field: value}; None removes the field) to the config.json entry named model_name.
    With create, a missing entry is added; otherwise nothing is written for it. Returns True if the file was written.
    """
    config_path = _get_config_path()
    with _config_write_lock:
        try:
            config = _read_config_file(config_path)
        except Exception as e:
            # Never overwrite a config file we could not parse
            logger.error(f"Not saving '{model_name}' to '{config_path}': could not read existing configuration: {e}")
            return False

        models = config.setdefault(model_type_key, [])
        entry = next((m for m in models if m.get("name") == model_name), None)
        created = entry is None
        if created:
            if not create:
                return False
            entry = {"name": model_name}
            models.append(entry)
        for field, value in changes.items():
            if value is None:
                entry.pop(field, None)
            else:
                entry[field] = value

        try:
            _write_config_file(config_path, config)
        except Exception as e:
            logger.error(f"Error writing '{config_path}': {e}")
            return False

    logger.info(f"{'Added' if created else 'Updated'} '{model_name}' in '{model_type_key}' of {os.path.basename(config_path)}")
    server.PromptServer.instance.send_sync("ondemand_loaders.config_updated", {"key": model_type_key, "name": model_name, "url": entry.get("url")})
    return True

def _get_api_key_for_url(model_url, api_key_param):
    """
    Determines the API key to use based on the model_url.
    It checks for a provided api_key_param first, then the site's environment variable (see sources.py).
    """
    source = get_source(model_url)
    return source.get_api_key(api_key_param) if source else api_key_param


def _request_model(model_url, model_name, api_key):
    """
    Starts the download request for a model and returns the streaming response. model_name may be None if not known yet.
    Raises RuntimeError with a message for the UI if the server refuses or fails the request.
    """
    subject = f"'{model_name}'" if model_name else "the model"
    headers = None
    if api_key:
        logger.info(f"Using provided API key")
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

    try:
        response = requests.get(model_url, stream=True, allow_redirects=True, headers=headers, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.HTTPError as e:
        message = f"Could not download {subject} from {model_url}: HTTP {e.response.status_code} {e.response.reason}."
        if e.response.status_code in (401, 403):
            message += " " + _auth_hint(model_url, api_key)
        raise _download_error(message) from e
    except requests.exceptions.RequestException as e:
        raise _download_error(f"Could not download {subject} from {model_url}: {e}") from e

    # Sites answer some unauthenticated downloads with a login page instead of an error status
    if response.headers.get('Content-Type', '').startswith('text/html'):
        response.close()
        raise _download_error(f"Could not download {subject} from {model_url}: the server returned a web page instead of a model file. "
                              + _auth_hint(model_url, api_key))
    return response

def _get_server_filename(model_url, response):
    """
    Returns the filename the server provides for a download, or None if it provides none or an unsafe one.
    """
    filename = server_filename(model_url, response, MODEL_EXTENSIONS)
    if filename:
        problem = check_server_filename(filename)
        if problem:
            logger.warning(f"Ignoring the file name '{filename}' sent by {model_url}: {problem}")
            return None
    return filename

def _download_model(model_url, model_name, destination_dir, api_key, download_chunks, recorded_sha256=None, target_name=None):
    """
    Handles the download of a model from a given URL to a specified directory.

    Args:
        model_url (str): The URL of the model to download.
        model_name (str): The name of the model (for logging purposes).
        destination_dir (str): The directory where the model should be saved.
        api_key (str): API key for authentication, if required.
        download_chunks (int): The size of download chunks in KB.
        recorded_sha256 (str): SHA256 recorded for the model in config.json by an earlier download, if any.
            The download is verified against it, unless the site publishes the file's hash (see sources.py).
        target_name (str): Model name to store the file under (see model_paths.py); it must pass check_model_name.
            If None, the file is stored under the filename the server provides.

    Returns:
        tuple: The full path to the model file, and the SHA256 of the file if it was downloaded
            (None if the file already existed or the model is marked offline).

    Raises:
        RuntimeError: If the model could not be downloaded, or does not match the expected SHA256;
            the message is shown in the ComfyUI UI.
    """
    if model_url == 'offline':
        logger.info(f"'{model_name}' is marked as offline. Assuming local file exists and skipping download.")
        # The user expects the model to exist, so we return the assumed path.
        return os.path.join(destination_dir, model_name), None

    response = _request_model(model_url, model_name, api_key)
    model_filename = _get_server_filename(model_url, response)

    if target_name:
        extension = model_extension(target_name, MODEL_EXTENSIONS) or (os.path.splitext(model_filename)[1] if model_filename else "")
        if not extension:
            response.close()
            raise _download_error(f"Could not download '{model_name}' from {model_url}: the server did not provide a file name, so the file type is unknown. "
                                  f"Add the file's extension to the model name, for example '{target_name}.safetensors'.")
        model_filepath = named_model_path(destination_dir, target_name, extension, MODEL_EXTENSIONS)

        # Models downloaded before files were named after their model are stored under the server's filename.
        # Use such a file only if its size matches, since servers give unrelated models the same name (e.g. model.safetensors).
        legacy_filepath = os.path.join(destination_dir, model_filename) if model_filename else None
        if legacy_filepath and legacy_filepath != model_filepath and os.path.isfile(legacy_filepath) and not os.path.exists(model_filepath):
            total_size = int(response.headers.get('content-length', 0))
            if not total_size or os.path.getsize(legacy_filepath) == total_size:
                logger.info(f"Using '{model_filename}', downloaded for '{model_name}' before files were named after their model.")
                response.close()
                return legacy_filepath, None
    else:
        if not model_filename:
            response.close()
            raise _download_error(f"Could not download '{model_name}' from {model_url}: the server did not provide a usable file name.")
        model_filepath = os.path.join(destination_dir, model_filename)

    if os.path.exists(model_filepath):
        logger.info(f"File '{model_filepath}' already exists. Skipping download.")
        response.close()
        return model_filepath, None
    else:
        os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
        expected_sha256, expected_from = _get_expected_sha256(model_url, response, model_filename, model_name, api_key, recorded_sha256)
        logger.info(f"Downloading '{model_name}' from '{model_url}' to '{model_filepath}'")
        # Download to a temporary file, so an interrupted download is not mistaken for a complete model next time
        part_filepath = model_filepath + ".part"
        try:
            total_size = int(response.headers.get('content-length', 0))
            block_size = (download_chunks or 4) * 1024
            # Progress bar on the node in the ComfyUI UI; without a known size there is no progress to show
            ui_progress_bar = comfy.utils.ProgressBar(total_size) if total_size else None
            downloaded = 0
            sha256 = hashlib.sha256()
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"{LOG_PREFIX} Downloading {model_name}") as progress_bar:
                with open(part_filepath, 'wb') as f:
                    for data in response.iter_content(block_size):
                        # Stop when the run is cancelled in the UI
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        progress_bar.update(len(data))
                        f.write(data)
                        sha256.update(data)
                        downloaded += len(data)
                        if ui_progress_bar:
                            ui_progress_bar.update_absolute(downloaded)
            downloaded_size = os.path.getsize(part_filepath)
            # content-length is the compressed size when the response is encoded, so it can only be checked otherwise
            if total_size and not response.headers.get('Content-Encoding') and downloaded_size != total_size:
                raise IOError(f"received {downloaded_size} of {total_size} bytes")
            digest = sha256.hexdigest()
            if expected_sha256 and digest != expected_sha256:
                message = f"the file's SHA256 {digest} does not match {expected_sha256} {expected_from}."
                if expected_from == "recorded in config.json":
                    message += " If the file was intentionally replaced on the server, remove 'sha256' from its entry in config.json."
                raise IOError(message)
            os.replace(part_filepath, model_filepath)
            logger.info(f"Successfully downloaded '{model_name}' to '{model_filepath}'"
                        + (f" (SHA256 verified {expected_from})." if expected_sha256 else "."))
            return model_filepath, digest
        except BaseException as e:
            # Also clean up when cancelled: ComfyUI's InterruptProcessingException is a BaseException
            response.close()
            try:
                os.remove(part_filepath)
            except OSError:
                pass
            if isinstance(e, comfy.model_management.InterruptProcessingException):
                logger.info(f"Download of '{model_name}' cancelled.")
            if not isinstance(e, Exception):
                raise # Let cancellation and other non-errors through unchanged
            raise _download_error(f"Download of '{model_name}' from {model_url} failed: {e}") from e

def _auth_hint(model_url, api_key):
    """
    Explains how to authenticate a download that was refused.
    """
    source = get_source(model_url)
    if source:
        return source.auth_hint(api_key)
    if api_key:
        return "The API key was rejected; check the 'api_key' field."
    return "The model may require an API key: set the 'api_key' field."

def _get_expected_sha256(model_url, response, model_filename, model_name, api_key, recorded_sha256):
    """
    Returns the SHA256 a download should have, and where it came from (for messages): the hash the site
    publishes if available, else the one recorded in config.json by an earlier download, else (None, None).
    """
    published_sha256 = None
    source = get_source(model_url)
    if source:
        try:
            published_sha256 = source.expected_sha256(model_url, response, model_filename, api_key)
        except Exception as e:
            logger.warning(f"Could not get the published SHA256 of '{model_name}' from {source.name}: {e}")

    if published_sha256:
        if recorded_sha256 and recorded_sha256 != published_sha256:
            logger.warning(f"The SHA256 {source.name} publishes for '{model_name}' differs from the one recorded in config.json; "
                           f"the file has changed on {source.name}, so the published one is used.")
        return published_sha256, f"published by {source.name}"
    if recorded_sha256:
        return recorded_sha256, "recorded in config.json"
    return None, None

def _download_error(message):
    """
    Logs a download failure and returns the exception to raise, which stops the workflow and shows the message in the UI.
    """
    logger.error(message)
    return RuntimeError(f"{LOG_PREFIX} {message}")

def _get_config_entry(model_name, model_type_key):
    """
    Returns the config.json entry for a model name (re-read so edits are picked up without a restart), or None.
    """
    return next((model for model in load_config(quiet=True).get(model_type_key, []) if model["name"] == model_name), None)

def _get_model_url_from_config(model_name, model_type_key, log_missing=True):
    """
    Retrieves the URL for a given model name from config.json.
    """
    model_url = (_get_config_entry(model_name, model_type_key) or {}).get("url")
    if not model_url and log_missing:
        logger.error(f"Model URL not found for name: {model_name} in {model_type_key}")
    return model_url

def _resolve_model_url(model_name, model_type_key, url):
    """
    Returns the URL to download model_name from. A URL entered in the node takes precedence, and is
    saved to config.json if it is new or differs from the configured one; otherwise config.json is used.
    """
    url = (url or "").strip()
    if url == LOCAL_URL:
        url = "" # Left over from selecting a local file
    if url and model_name != "None":
        if _get_model_url_from_config(model_name, model_type_key, log_missing=False) != url:
            # A recorded hash belongs to the old URL's file
            _update_config_entry(model_type_key, model_name, {"url": url, "sha256": None}, create=True)
        return url
    return _get_model_url_from_config(model_name, model_type_key)

def _local_files(model_type_key):
    """
    Returns the model files ComfyUI finds for a config.json list, as paths relative to their folder with / separators.
    """
    folder_key = MODEL_FOLDER_KEYS[model_type_key]
    if folder_key not in folder_paths.folder_names_and_paths:
        return []
    return [path.replace("\\", "/") for path in folder_paths.get_filename_list(folder_key)]

def _model_choices(model_type_key):
    """
    The drop-down entries for a config.json list: its models, then the local model files that do not belong to any of them.
    A model's file is named after it (see model_paths.py): the name itself, or the name plus a model extension.
    """
    names = [model["name"] for model in load_config().get(model_type_key, [])]
    owned = {os.path.normcase(name) for name in names}
    local = [LOCAL_PREFIX + path for path in _local_files(model_type_key)
             if os.path.normcase(path) not in owned and os.path.normcase(os.path.splitext(path)[0]) not in owned]
    return names + local

def _local_model_path(model_name):
    """
    Returns the file path of a local drop-down entry ("[local] <path>"), or None if model_name is not one.
    """
    return model_name[len(LOCAL_PREFIX):] if model_name.startswith(LOCAL_PREFIX) else None

def _local_model_problem(local_path, model_type_key):
    """
    Returns why a local model file cannot be loaded, or None if it can.
    """
    folder_key = MODEL_FOLDER_KEYS[model_type_key]
    if folder_key not in folder_paths.folder_names_and_paths:
        return f"local {MODEL_DIRS[model_type_key]} models are not available"
    if not folder_paths.get_full_path(folder_key, local_path):
        return f"the file '{local_path}' was not found in models/{MODEL_DIRS[model_type_key]}"
    return None

def _get_model_file(model_name, model_type_key, url, api_key, download_chunks):
    """
    Downloads model_name if needed and returns its path within its models folder (MODEL_DIRS), as ComfyUI's loaders take it.
    The file is named after the model (see model_paths.py), unless the name cannot be used as a path; such names,
    which can only come from editing config.json by hand, keep the filename the server provides.
    Raises an exception, stopping the workflow, if there is no model to load.
    """
    if model_name == "None":
        raise ValueError(f"{LOG_PREFIX} No model selected ({model_type_key})")

    # Local files are loaded as they are; the url field does not apply to them
    local_path = _local_model_path(model_name)
    if local_path is not None:
        problem = _local_model_problem(local_path, model_type_key)
        if problem:
            raise ValueError(f"{LOG_PREFIX} Cannot load '{model_name}': {problem}")
        return local_path

    model_url = _resolve_model_url(model_name, model_type_key, url)
    if not model_url:
        raise ValueError(f"{LOG_PREFIX} '{model_name}' has no download URL in config.json ({model_type_key}); enter one in the url field")

    destination_dir = os.path.join(folder_paths.models_dir, MODEL_DIRS[model_type_key])
    problem = check_model_name(model_name)
    target_name = None if problem else model_name
    if problem:
        logger.info(f"Storing '{model_name}' under the server's file name, since the model name cannot be used as one: {problem}")
    elif model_url != "offline":
        # Already downloaded: no need to contact the server
        existing_filepath = find_named_model_file(destination_dir, model_name, MODEL_EXTENSIONS)
        if existing_filepath:
            return relative_model_path(destination_dir, existing_filepath)

    api_key = _get_api_key_for_url(model_url, api_key)
    recorded_sha256 = (_get_config_entry(model_name, model_type_key) or {}).get("sha256")
    model_filepath, sha256 = _download_model(model_url, model_name, destination_dir, api_key, download_chunks, recorded_sha256, target_name)
    if sha256 and sha256 != recorded_sha256:
        _update_config_entry(model_type_key, model_name, {"sha256": sha256})
    return relative_model_path(destination_dir, model_filepath)

def _validate_model_choice(model_name, model_type_key, url):
    """
    VALIDATE_INPUTS helper: the model name may be any name in config.json, a local model file, or a new name if a URL is given.
    """
    if model_name == "None" or _get_model_url_from_config(model_name, model_type_key, log_missing=False):
        return True
    local_path = _local_model_path(model_name)
    if local_path is not None:
        problem = _local_model_problem(local_path, model_type_key)
        return f"Cannot load '{model_name}': {problem}" if problem else True
    if (url or "").strip() in ("", LOCAL_URL):
        return f"'{model_name}' is not in config.json ({model_type_key}); enter its download URL in the url field"
    problem = check_model_name(model_name)
    if problem:
        return f"'{model_name}' cannot be used as a model name: {problem}"
    return True

def _check_new_model(model_type_key, url, subfolder, name, api_key):
    """
    Checks a model about to be added with the New model dialog, and returns (full name, None) or (None, error message).
    The full name is "<subfolder>/<name>", or just the name without a subfolder. A blank name is replaced by the
    filename the server provides for url, which requires a request to the server.
    """
    destination_dir = os.path.join(folder_paths.models_dir, MODEL_DIRS[model_type_key])
    if not url:
        return None, "Enter the model's download URL."

    subfolder = normalize_subfolder(subfolder)
    if subfolder:
        problem = check_subfolder(subfolder)
        if problem:
            return None, f"The subfolder '{subfolder}' cannot be used: {problem}."
    if "/" in name or "\\" in name:
        return None, "Use the Subfolder field to put the model in a folder; the name cannot contain / or \\."
    in_subfolder = lambda filename: f"{subfolder}/{filename}" if subfolder else filename

    if not name:
        if url == "offline":
            return None, "Enter a name for an offline model: the name of its file."
        try:
            response = _request_model(url, None, _get_api_key_for_url(url, api_key))
        except RuntimeError as e:
            return None, str(e).removeprefix(LOG_PREFIX).strip()
        server_name = _get_server_filename(url, response)
        response.close()
        if not server_name:
            return None, ("The server did not provide a usable file name for this URL. "
                          "Enter a name for the model, including the file's extension (for example .safetensors).")
        name = in_subfolder(server_name)
        entry = _get_config_entry(name, model_type_key)
        if entry and entry.get("url") != url:
            return None, f"A model named '{name}' is already in config.json with a different URL. Enter a name for this one."
        # A file under the server's own filename is most likely this model, downloaded before
        return name, None

    name = in_subfolder(name)
    problem = check_model_name(name)
    if problem:
        return None, f"'{name}' cannot be used as a model name: {problem}."
    if name.startswith(LOCAL_PREFIX):
        return None, f"Model names cannot start with '{LOCAL_PREFIX.strip()}', which marks local files in the drop-down."
    # Entering the name of a model in config.json updates its URL; a new name must not match an existing file,
    # which would be loaded instead of downloading the model
    existing_filepath = None if _get_config_entry(name, model_type_key) else find_named_model_file(destination_dir, name, MODEL_EXTENSIONS)
    if existing_filepath:
        return None, (f"A model file named '{name}' already exists in models/{MODEL_DIRS[model_type_key]}. Choose another name, "
                      f"or cancel and select '{LOCAL_PREFIX}{relative_model_path(destination_dir, existing_filepath)}' to use the file.")
    return name, None

def _core_input(node_class, input_name):
    """
    Returns a core loader node's definition of one of its inputs (such as CLIPLoader's "type"), so the options
    offered here always match the ComfyUI version in use.
    """
    input_types = node_class.INPUT_TYPES()
    return input_types.get("required", {}).get(input_name) or input_types.get("optional", {})[input_name]

URL_TOOLTIP = "Download URL of the selected model, filled in from config.json. Enter a new name and URL to add the model to config.json, or change the URL to update it."

def _url_input(model_type_key, name_input):
    """
    The url widget. The ondemand_* options tell js/onDemandLoader.js which name widget and config.json list it belongs to.
    """
    return ("STRING", {"default": "", "multiline": False, "tooltip": URL_TOOLTIP,
                       "ondemand_config_key": model_type_key, "ondemand_name_input": name_input})


@server.PromptServer.instance.routes.get("/on_demand_loader/model_url")
async def model_url_handler(request):
    model_name = request.query.get("name") or ""
    if _local_model_path(model_name) is not None:
        return web.json_response({"url": LOCAL_URL})
    model_url = _get_model_url_from_config(model_name, request.query.get("key"), log_missing=False)
    return web.json_response({"url": model_url or ""})

@server.PromptServer.instance.routes.post("/on_demand_loader/new_model")
async def new_model_handler(request):
    data = await request.json()
    model_type_key = data.get("key")
    if model_type_key not in MODEL_DIRS:
        return web.json_response({"error": f"Unknown model list '{model_type_key}'"}, status=400)
    # Looking up the server's filename makes a blocking request, so keep it off the event loop
    name, error = await asyncio.to_thread(_check_new_model, model_type_key, (data.get("url") or "").strip(),
                                          data.get("subfolder") or "", (data.get("name") or "").strip(), data.get("api_key") or None)
    return web.json_response({"error": error} if error else {"name": name})

@server.PromptServer.instance.routes.get("/on_demand_loader/subfolders")
async def subfolders_handler(request):
    model_type_key = request.query.get("key")
    if model_type_key not in MODEL_DIRS:
        return web.json_response({"error": f"Unknown model list '{model_type_key}'"}, status=400)
    models_dir = os.path.join(folder_paths.models_dir, MODEL_DIRS[model_type_key])
    return web.json_response({"subfolders": await asyncio.to_thread(list_subfolders, models_dir)})

@server.PromptServer.instance.routes.get("/on_demand_loader/export")
async def export_handler(request):
    """
    Returns the config.json sections listed in ?keys=a,b (all model sections without it), for saving to a file.
    """
    keys = [key for key in request.query.get("keys", "").split(",") if key] or None
    if keys and any(key not in MODEL_DIRS for key in keys):
        return web.json_response({"error": f"Unknown model lists: {', '.join(k for k in keys if k not in MODEL_DIRS)}"}, status=400)
    try:
        config = _read_config_file(_get_config_path())
    except Exception as e:
        return web.json_response({"error": f"Could not read config.json: {e}"}, status=500)
    if keys is None:
        keys = [key for key in config if key in MODEL_DIRS]
    return web.json_response(export_sections(config, keys))

MAX_IMPORT_SIZE = 10 * 1024 * 1024

@server.PromptServer.instance.routes.post("/on_demand_loader/import")
async def import_handler(request):
    """
    Merges an exported configuration into config.json. Body: {"config": <exported file>, "apply": bool, "use_imported": bool}.
    Without apply, only reports what would change. Conflicts (same name, different URL) keep the existing entry
    unless use_imported. config.json is saved as config.json.bak before it is changed.
    """
    if (request.content_length or 0) > MAX_IMPORT_SIZE:
        return web.json_response({"error": "The file is too large to be an exported configuration."}, status=413)
    data = await request.json()
    imported, apply, use_imported = data.get("config"), bool(data.get("apply")), bool(data.get("use_imported"))
    config_path = _get_config_path()

    with _config_write_lock:
        try:
            config = _read_config_file(config_path)
        except Exception as e:
            return web.json_response({"error": f"config.json could not be read, so nothing can be imported into it: {e}"}, status=500)
        plan = plan_import(config, imported, MODEL_DIRS.keys(), LOCAL_PREFIX)
        summary = {"sections": plan["sections"], "skipped_sections": plan["skipped_sections"],
                   "changes": {"keep": count_changes(plan, False), "imported": count_changes(plan, True)}}
        if not apply:
            return web.json_response(summary)

        changes = count_changes(plan, use_imported)
        if changes:
            try:
                if os.path.exists(config_path):
                    shutil.copy2(config_path, config_path + ".bak")
                _write_config_file(config_path, apply_import(config, plan, use_imported))
            except Exception as e:
                logger.error(f"Error writing '{config_path}': {e}")
                return web.json_response({"error": f"Could not write config.json: {e}"}, status=500)
            logger.info(f"Imported {changes} model change(s) into {os.path.basename(config_path)}; the previous version is in {os.path.basename(config_path)}.bak")
    return web.json_response({**summary, "applied": changes})


class OnDemandLoraLoader:

    @classmethod
    def INPUT_TYPES(cls):

        loras = _model_choices("loras")

        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (loras,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "clip": ("CLIP", ),
                "api_key": ("STRING", {"default": None, "multiline": False}),
                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "url": _url_input("loras", "lora_name")
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, lora_name, url=None):
        return _validate_model_choice(lora_name, "loras", url)

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "download_lora"
    DESCRIPTION = "Load loras models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"

    CATEGORY = "loaders"

    def download_lora(self, model, lora_name, strength_model, strength_clip, clip=None, api_key=None, download_chunks=None, url=None):
        self.lora_loader = LoraLoader()

        if lora_name == "None":
            return model, clip # No lora selected: pass model/clip through unchanged

        lora_filename = _get_model_file(lora_name, "loras", url, api_key, download_chunks)

        # Load the LORA using the existing LoraLoader
        model_lora, clip_lora = self.lora_loader.load_lora(model, clip, lora_filename, strength_model, strength_clip)
        return model_lora, clip_lora


class OnDemandUNETLoader:

    @classmethod
    def INPUT_TYPES(cls):

        models = _model_choices("diffusion_models")

        return {
            "required": {
                "unet_name": (models,),
                "weight_dtype": _core_input(UNETLoader, "weight_dtype")
            },
            "optional": {
                "api_key": ("STRING", {"default": None, "multiline": False}),
                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "url": _url_input("diffusion_models", "unet_name")
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, unet_name, url=None):
        return _validate_model_choice(unet_name, "diffusion_models", url)

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "download_unet"
    DESCRIPTION = "Load diffusion models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"

    CATEGORY = "loaders"

    def download_unet(self, unet_name, weight_dtype, api_key=None, download_chunks=None, url=None):
        self.unet_loader = UNETLoader()

        model_filename = _get_model_file(unet_name, "diffusion_models", url, api_key, download_chunks)

        # Load the Model using the existing UNETLoader
        model_output = self.unet_loader.load_unet(model_filename, weight_dtype)
        return model_output


class OnDemandCheckpointLoader:

    @classmethod
    def INPUT_TYPES(cls):

        models = _model_choices("checkpoints")

        return {
            "required": {
                "ckpt_name": (models,)
            },
            "optional": {
                "api_key": ("STRING", {"default": None, "multiline": False}),
                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "url": _url_input("checkpoints", "ckpt_name")
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, ckpt_name, url=None):
        return _validate_model_choice(ckpt_name, "checkpoints", url)

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "download_checkpoint"
    DESCRIPTION = "Load checkpoint models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"
    CATEGORY = "loaders"

    def download_checkpoint(self, ckpt_name, api_key=None, download_chunks=None, url=None):
        self.checkpoint_loader = CheckpointLoaderSimple()

        model_filename = _get_model_file(ckpt_name, "checkpoints", url, api_key, download_chunks)

        # Load the checkpoint using the existing CheckpointLoaderSimple
        return self.checkpoint_loader.load_checkpoint(model_filename)

class OnDemandVAELoader:

    @classmethod
    def INPUT_TYPES(cls):

        models = _model_choices("vae_models")

        return {
            "required": {
                "vae_name": (models,)
            },
            "optional": {
                "api_key": ("STRING", {"default": None, "multiline": False}),
                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "url": _url_input("vae_models", "vae_name")
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, vae_name, url=None):
        return _validate_model_choice(vae_name, "vae_models", url)

    RETURN_TYPES = ("VAE",)
    FUNCTION = "download_vae"
    DESCRIPTION = "Load vae models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"
    CATEGORY = "loaders"

    def download_vae(self, vae_name, api_key=None, download_chunks=None, url=None):
        self.vae_loader = VAELoader()

        model_filename = _get_model_file(vae_name, "vae_models", url, api_key, download_chunks)

        # Load vae using the existing VAELoader
        return self.vae_loader.load_vae(model_filename)

class OnDemandCLIPLoader:

    @classmethod
    def INPUT_TYPES(s):

        models = _model_choices("clip_models")

        return {"required": {
                                "clip_name": (models,),
                                "type": _core_input(CLIPLoader, "type"),
                              },
                "optional": {
                                "device": (["default", "cpu"], {"advanced": True}),
                                "api_key": ("STRING", {"default": None, "multiline": False}),
                                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                                "url": _url_input("clip_models", "clip_name")
                             }}

    @classmethod
    def VALIDATE_INPUTS(cls, clip_name, url=None):
        return _validate_model_choice(clip_name, "clip_models", url)

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "download_clip"
    CATEGORY = "loaders"
    DESCRIPTION = "Load clip models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"

    def download_clip(self, clip_name, type="stable_diffusion", device="default", api_key=None, download_chunks=None, url=None):
        self.clip_loader = CLIPLoader()

        model_filename = _get_model_file(clip_name, "clip_models", url, api_key, download_chunks)

        # Load the checkpoint using the existing CheckpointLoaderSimple
        return self.clip_loader.load_clip(model_filename, type, device)


class OnDemandDualCLIPLoader:

    @classmethod
    def INPUT_TYPES(s):

        models = _model_choices("clip_models")

        return {"required": {
                                "clip_name1": (models,),
                                "clip_name2": (models,),
                                "type": _core_input(DualCLIPLoader, "type"),
                              },
                "optional": {
                                "device": (["default", "cpu"], {"advanced": True}),
                                "api_key": ("STRING", {"default": None, "multiline": False}),
                                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                                "url1": _url_input("clip_models", "clip_name1"),
                                "url2": _url_input("clip_models", "clip_name2")
                             }}

    @classmethod
    def VALIDATE_INPUTS(cls, clip_name1, clip_name2, url1=None, url2=None):
        result = _validate_model_choice(clip_name1, "clip_models", url1)
        if result is not True:
            return result
        return _validate_model_choice(clip_name2, "clip_models", url2)

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "download_clip"
    CATEGORY = "loaders"
    DESCRIPTION = "Load (dual) clip models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"

    def download_clip(self, clip_name1, clip_name2, type, device="default", api_key=None, download_chunks=None, url1=None, url2=None):
        self.clip_loader = DualCLIPLoader()

        model_filename1 = _get_model_file(clip_name1, "clip_models", url1, api_key, download_chunks)
        model_filename2 = _get_model_file(clip_name2, "clip_models", url2, api_key, download_chunks)

        return self.clip_loader.load_clip(model_filename1, model_filename2, type, device)

class OnDemandCLIPVisionLoader:

    @classmethod
    def INPUT_TYPES(s):

        models = _model_choices("clip_vision")

        return {"required": {
                                "clip_name": (models,),
                              },
                "optional": {
                                "api_key": ("STRING", {"default": None, "multiline": False}),
                                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                                "url": _url_input("clip_vision", "clip_name")
                             }}

    @classmethod
    def VALIDATE_INPUTS(cls, clip_name, url=None):
        return _validate_model_choice(clip_name, "clip_vision", url)

    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "download_clip"
    CATEGORY = "loaders"
    DESCRIPTION = "Load clip vision models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"

    def download_clip(self, clip_name, api_key=None, download_chunks=None, url=None):
        self.clip_loader = CLIPVisionLoader()

        model_filename = _get_model_file(clip_name, "clip_vision", url, api_key, download_chunks)

        return self.clip_loader.load_clip(model_filename)


class OnDemandGGUFLoader:

    @classmethod
    def INPUT_TYPES(s):

        models = _model_choices("gguf_models")

        return {"required": {
                                "unet_name": (models,)
                              },
                "optional": {
                                "api_key": ("STRING", {"default": None, "multiline": False}),
                                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                                "url": _url_input("gguf_models", "unet_name")
                             }}

    @classmethod
    def VALIDATE_INPUTS(cls, unet_name, url=None):
        return _validate_model_choice(unet_name, "gguf_models", url)

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "download_unet"
    CATEGORY = "loaders"
    DESCRIPTION = "Load gguf models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"

    def download_unet(self, unet_name, api_key=None, download_chunks=None, url=None):
        if module_gguf is None:
            raise RuntimeError(f"{LOG_PREFIX} UnetLoaderGGUF class not available. Ensure ComfyUI-GGUF is installed correctly.")

        self.gguf_loader = module_gguf.nodes.UnetLoaderGGUF()

        model_filename = _get_model_file(unet_name, "gguf_models", url, api_key, download_chunks)

        # Load the gguf using the existing UnetLoaderGGUF
        return self.gguf_loader.load_unet(model_filename)

class OnDemandControlNetLoader:

    @classmethod
    def INPUT_TYPES(cls):

        models = _model_choices("controlnet_models")

        return {
            "required": {
                "control_net_name": (models,)
            },
            "optional": {
                "api_key": ("STRING", {"default": None, "multiline": False}),
                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "url": _url_input("controlnet_models", "control_net_name")
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, control_net_name, url=None):
        return _validate_model_choice(control_net_name, "controlnet_models", url)

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "download_controlnet"
    DESCRIPTION = "Load control_net models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"
    CATEGORY = "loaders"

    def download_controlnet(self, control_net_name, api_key=None, download_chunks=None, url=None):
        self.controlnet_loader = ControlNetLoader()

        model_filename = _get_model_file(control_net_name, "controlnet_models", url, api_key, download_chunks)

        # Load vae using the existing VAELoader
        return self.controlnet_loader.load_controlnet(model_filename)


class OnDemandModelPatchLoader:

    @classmethod
    def INPUT_TYPES(cls):

        models = _model_choices("model_patches")

        return {
            "required": {
                "name": (models,)
            },
            "optional": {
                "api_key": ("STRING", {"default": None, "multiline": False}),
                "download_chunks": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "url": _url_input("model_patches", "name")
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, name, url=None):
        return _validate_model_choice(name, "model_patches", url)

    RETURN_TYPES = ("MODEL_PATCH",)
    FUNCTION = "download_model_patch"
    DESCRIPTION = "Load model_patch models from CivitAI/HuggingFace, they will be downloaded automatically if not found.\nPut a valid CivitAI/HuggingFace API key in form field 'api_key' or in CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable to access private models"
    CATEGORY = "loaders"

    def download_model_patch(self, name, api_key=None, download_chunks=None, url=None):
        self.model_patch_loader = ModelPatchLoader()

        model_filename = _get_model_file(name, "model_patches", url, api_key, download_chunks)

        # Load model_patch using the existing ModelPatchLoader
        return self.model_patch_loader.load_model_patch(model_filename)
