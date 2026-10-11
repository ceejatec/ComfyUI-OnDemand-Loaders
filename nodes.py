import json
import os
import requests
import sys
import logging
import threading
from tqdm import tqdm
import re
import folder_paths
import comfy.utils
import comfy.model_management
from pathlib import Path
import importlib.util
from urllib.parse import urlparse
import server
from aiohttp import web
from nodes import LoraLoader, UNETLoader, CheckpointLoaderSimple, VAELoader, CLIPLoader,  ControlNetLoader, DualCLIPLoader, CLIPVisionLoader
from comfy_extras.nodes_model_patch import ModelPatchLoader


LOG_PREFIX = "[ComfyUI-OnDemand-Loaders]"


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

def _save_model_to_config(model_type_key, model_name, model_url):
    """
    Adds a model entry to config.json, or updates the URL of an existing entry with the same name.
    Returns True if the file was written.
    """
    config_path = _get_config_path()
    with _config_write_lock:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        except Exception as e:
            # Never overwrite a config file we could not parse
            logger.error(f"Not saving '{model_name}' to '{config_path}': could not read existing configuration: {e}")
            return False

        models = config.setdefault(model_type_key, [])
        entry = next((m for m in models if m.get("name") == model_name), None)
        if entry:
            entry["url"] = model_url
        else:
            models.append({"name": model_name, "url": model_url})

        try:
            tmp_path = config_path + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(config, f, indent=4)
                f.write("\n")
            os.replace(tmp_path, config_path)
        except Exception as e:
            logger.error(f"Error writing '{config_path}': {e}")
            return False

    logger.info(f"{'Updated' if entry else 'Added'} '{model_name}' in '{model_type_key}' of {os.path.basename(config_path)}")
    server.PromptServer.instance.send_sync("ondemand_loaders.config_updated", {"key": model_type_key, "name": model_name, "url": model_url})
    return True

CIVITAI_DOMAINS = ("civitai.com", "civitai.red", "civitai.green")
HUGGINGFACE_DOMAINS = ("huggingface.co", "hf.co")

def _url_platform(model_url):
    """
    Returns "civitai" or "huggingface" if model_url is hosted on one of their domains (or subdomains), else None.
    """
    host = (urlparse(model_url).hostname or "").lower()
    for platform, domains in (("civitai", CIVITAI_DOMAINS), ("huggingface", HUGGINGFACE_DOMAINS)):
        if any(host == domain or host.endswith("." + domain) for domain in domains):
            return platform
    return None

def _get_api_key_for_url(model_url, api_key_param):
    """
    Determines the API key to use based on the model_url.
    It checks for a provided api_key_param first, then environment variables.
    """
    platform = _url_platform(model_url)
    if platform == "civitai":
        return api_key_param or os.environ.get('CIVITAI_TOKEN')
    elif platform == "huggingface":
        return api_key_param or os.environ.get('HUGGINGFACE_TOKEN')
    else:
        return api_key_param # Return provided key if URL doesn't match known platforms


def _download_model(model_url, model_name, destination_dir, api_key, download_chunks):
    """
    Handles the download of a model from a given URL to a specified directory.

    Args:
        model_url (str): The URL of the model to download.
        model_name (str): The name of the model (for logging purposes).
        destination_dir (str): The directory where the model should be saved.
        api_key (str): API key for authentication, if required.
        download_chunks (int): The size of download chunks in KB.

    Returns:
        str: The full path to the downloaded model file.

    Raises:
        RuntimeError: If the model could not be downloaded; the message is shown in the ComfyUI UI.
    """
    if model_url == 'offline':
        logger.info(f"'{model_name}' is marked as offline. Assuming local file exists and skipping download.")
        # The user expects the model to exist, so we return the assumed path.
        return os.path.join(destination_dir, model_name)

    os.makedirs(destination_dir, exist_ok=True)

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
        message = f"Could not download '{model_name}' from {model_url}: HTTP {e.response.status_code} {e.response.reason}."
        if e.response.status_code in (401, 403):
            message += " " + _auth_hint(model_url, api_key)
        raise _download_error(message) from e
    except requests.exceptions.RequestException as e:
        raise _download_error(f"Could not download '{model_name}' from {model_url}: {e}") from e

    # Sites answer some unauthenticated downloads with a login page instead of an error status
    if response.headers.get('Content-Type', '').startswith('text/html'):
        raise _download_error(f"Could not download '{model_name}' from {model_url}: the server returned a web page instead of a model file. "
                              + _auth_hint(model_url, api_key))

    model_filename = None
    content_disposition = response.headers.get('Content-Disposition')
    if content_disposition:
        filename_match = re.search(r'filename="?([^"]+)"?', content_disposition)
        if filename_match:
            model_filename = filename_match.group(1).strip()

    if not model_filename:
        # Fallback to extracting filename from URL if Content-Disposition is missing or malformed
        model_filename = os.path.basename(model_url)

    model_filepath = os.path.join(destination_dir, model_filename)

    if os.path.exists(model_filepath):
        logger.info(f"File '{model_filename}' already exists at '{model_filepath}'. Skipping download.")
        return model_filepath
    else:
        logger.info(f"Downloading '{model_name}' from '{model_url}' to '{model_filepath}'")
        # Download to a temporary file, so an interrupted download is not mistaken for a complete model next time
        part_filepath = model_filepath + ".part"
        try:
            total_size = int(response.headers.get('content-length', 0))
            block_size = (download_chunks or 4) * 1024
            # Progress bar on the node in the ComfyUI UI; without a known size there is no progress to show
            ui_progress_bar = comfy.utils.ProgressBar(total_size) if total_size else None
            downloaded = 0
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"{LOG_PREFIX} Downloading {model_name}") as progress_bar:
                with open(part_filepath, 'wb') as f:
                    for data in response.iter_content(block_size):
                        # Stop when the run is cancelled in the UI
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        progress_bar.update(len(data))
                        f.write(data)
                        downloaded += len(data)
                        if ui_progress_bar:
                            ui_progress_bar.update_absolute(downloaded)
            downloaded_size = os.path.getsize(part_filepath)
            # content-length is the compressed size when the response is encoded, so it can only be checked otherwise
            if total_size and not response.headers.get('Content-Encoding') and downloaded_size != total_size:
                raise IOError(f"received {downloaded_size} of {total_size} bytes")
            os.replace(part_filepath, model_filepath)
            logger.info(f"Successfully downloaded '{model_name}' filename {model_filename}.")
            return model_filepath
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
    if api_key:
        return "The API key was rejected; check the 'api_key' field or the CIVITAI_TOKEN/HUGGINGFACE_TOKEN environment variable."
    platform = _url_platform(model_url)
    if platform == "civitai":
        return "The model may require a Civitai API key: set the CIVITAI_TOKEN environment variable or the 'api_key' field."
    if platform == "huggingface":
        return "The model may require a HuggingFace token: set the HUGGINGFACE_TOKEN environment variable or the 'api_key' field."
    return "The model may require an API key: set the 'api_key' field."

def _download_error(message):
    """
    Logs a download failure and returns the exception to raise, which stops the workflow and shows the message in the UI.
    """
    logger.error(message)
    return RuntimeError(f"{LOG_PREFIX} {message}")

def _get_model_url_from_config(model_name, model_type_key, log_missing=True):
    """
    Retrieves the URL for a given model name from config.json (re-read so edits are picked up without a restart).
    """
    model_url = None
    for model in load_config(quiet=True).get(model_type_key, []):
        if model["name"] == model_name:
            model_url = model["url"]
            break
    if not model_url and log_missing:
        logger.error(f"Model URL not found for name: {model_name} in {model_type_key}")
    return model_url

def _resolve_model_url(model_name, model_type_key, url):
    """
    Returns the URL to download model_name from. A URL entered in the node takes precedence, and is
    saved to config.json if it is new or differs from the configured one; otherwise config.json is used.
    """
    url = (url or "").strip()
    if url and model_name != "None":
        if _get_model_url_from_config(model_name, model_type_key, log_missing=False) != url:
            _save_model_to_config(model_type_key, model_name, url)
        return url
    return _get_model_url_from_config(model_name, model_type_key)

def _get_model_file(model_name, model_type_key, url, models_subdir, api_key, download_chunks):
    """
    Downloads model_name if needed and returns its filename in models/<models_subdir>.
    Raises an exception, stopping the workflow, if there is no model to load.
    """
    if model_name == "None":
        raise ValueError(f"{LOG_PREFIX} No model selected ({model_type_key})")

    model_url = _resolve_model_url(model_name, model_type_key, url)
    if not model_url:
        raise ValueError(f"{LOG_PREFIX} '{model_name}' has no download URL in config.json ({model_type_key}); enter one in the url field")

    destination_dir = os.path.join(folder_paths.models_dir, models_subdir)
    api_key = _get_api_key_for_url(model_url, api_key)
    return os.path.basename(_download_model(model_url, model_name, destination_dir, api_key, download_chunks))

def _validate_model_choice(model_name, model_type_key, url):
    """
    VALIDATE_INPUTS helper: the model name may be any name in config.json, or a new name if a URL is given.
    """
    if model_name == "None" or (url or "").strip() or _get_model_url_from_config(model_name, model_type_key, log_missing=False):
        return True
    return f"'{model_name}' is not in config.json ({model_type_key}); enter its download URL in the url field"

URL_TOOLTIP = "Download URL of the selected model, filled in from config.json. Enter a new name and URL to add the model to config.json, or change the URL to update it."

def _url_input(model_type_key, name_input):
    """
    The url widget. The ondemand_* options tell js/onDemandLoader.js which name widget and config.json list it belongs to.
    """
    return ("STRING", {"default": "", "multiline": False, "tooltip": URL_TOOLTIP,
                       "ondemand_config_key": model_type_key, "ondemand_name_input": name_input})


@server.PromptServer.instance.routes.get("/on_demand_loader/model_url")
async def model_url_handler(request):
    model_url = _get_model_url_from_config(request.query.get("name"), request.query.get("key"), log_missing=False)
    return web.json_response({"url": model_url or ""})


class OnDemandLoraLoader:

    @classmethod
    def INPUT_TYPES(cls):

        NODE_CONFIG = load_config()
        loras = [lora["name"] for lora in NODE_CONFIG.get("loras", []) ]

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

        lora_filename = _get_model_file(lora_name, "loras", url, "loras", api_key, download_chunks)

        # Load the LORA using the existing LoraLoader
        model_lora, clip_lora = self.lora_loader.load_lora(model, clip, lora_filename, strength_model, strength_clip)
        return model_lora, clip_lora


class OnDemandUNETLoader:

    @classmethod
    def INPUT_TYPES(cls):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("diffusion_models", []) ]

        return {
            "required": {
                "unet_name": (models,),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
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

        model_filename = _get_model_file(unet_name, "diffusion_models", url, "diffusion_models", api_key, download_chunks)

        # Load the Model using the existing UNETLoader
        model_output = self.unet_loader.load_unet(model_filename, weight_dtype)
        return model_output


class OnDemandCheckpointLoader:

    @classmethod
    def INPUT_TYPES(cls):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("checkpoints", []) ]

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

        model_filename = _get_model_file(ckpt_name, "checkpoints", url, "checkpoints", api_key, download_chunks)

        # Load the checkpoint using the existing CheckpointLoaderSimple
        return self.checkpoint_loader.load_checkpoint(model_filename)

class OnDemandVAELoader:

    @classmethod
    def INPUT_TYPES(cls):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("vae_models", []) ]

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

        model_filename = _get_model_file(vae_name, "vae_models", url, "vae", api_key, download_chunks)

        # Load vae using the existing VAELoader
        return self.vae_loader.load_vae(model_filename)

class OnDemandCLIPLoader:

    @classmethod
    def INPUT_TYPES(s):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("clip_models", []) ]

        return {"required": {
                                "clip_name": (models,),
                                "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image"], ),
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

        model_filename = _get_model_file(clip_name, "clip_models", url, "text_encoders", api_key, download_chunks)

        # Load the checkpoint using the existing CheckpointLoaderSimple
        return self.clip_loader.load_clip(model_filename, type, device)


class OnDemandDualCLIPLoader:

    @classmethod
    def INPUT_TYPES(s):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("clip_models", []) ]

        return {"required": {
                                "clip_name1": (models,),
                                "clip_name2": (models,),
                                "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream", "hunyuan_image", "hunyuan_video_15"], ),
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

        model_filename1 = _get_model_file(clip_name1, "clip_models", url1, "text_encoders", api_key, download_chunks)
        model_filename2 = _get_model_file(clip_name2, "clip_models", url2, "text_encoders", api_key, download_chunks)

        return self.clip_loader.load_clip(model_filename1, model_filename2, type, device)

class OnDemandCLIPVisionLoader:

    @classmethod
    def INPUT_TYPES(s):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("clip_vision", []) ]

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

        model_filename = _get_model_file(clip_name, "clip_vision", url, "clip_vision", api_key, download_chunks)

        return self.clip_loader.load_clip(model_filename)


class OnDemandGGUFLoader:

    @classmethod
    def INPUT_TYPES(s):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("gguf_models", []) ]

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

        model_filename = _get_model_file(unet_name, "gguf_models", url, "unet", api_key, download_chunks)

        # Load the gguf using the existing UnetLoaderGGUF
        return self.gguf_loader.load_unet(model_filename)

class OnDemandControlNetLoader:

    @classmethod
    def INPUT_TYPES(cls):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("controlnet_models", []) ]

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

        model_filename = _get_model_file(control_net_name, "controlnet_models", url, "controlnet", api_key, download_chunks)

        # Load vae using the existing VAELoader
        return self.controlnet_loader.load_controlnet(model_filename)


class OnDemandModelPatchLoader:

    @classmethod
    def INPUT_TYPES(cls):

        NODE_CONFIG = load_config()
        models = [model["name"] for model in NODE_CONFIG.get("model_patches", []) ]

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

        model_filename = _get_model_file(name, "model_patches", url, "model_patches", api_key, download_chunks)

        # Load model_patch using the existing ModelPatchLoader
        return self.model_patch_loader.load_model_patch(model_filename)
