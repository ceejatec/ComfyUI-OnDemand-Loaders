"""
Sites that models are downloaded from.

To support another site, subclass ModelSource, set its name, domains and token_env_var, override
expected_sha256 if the site publishes file hashes, and add an instance to MODEL_SOURCES.
"""
import os
import re
from urllib.parse import urlparse, parse_qs

import requests


class ModelSource:
    name = None           # Shown in messages, e.g. "Civitai"
    domains = ()          # Host names the site serves downloads from; subdomains match too
    token_env_var = None  # Environment variable holding the user's API token, if the site uses one

    def matches(self, url):
        host = (urlparse(url).hostname or "").lower()
        return any(host == domain or host.endswith("." + domain) for domain in self.domains)

    def get_api_key(self, api_key_param):
        """
        The API key to send: the one entered in the node, else the site's environment variable.
        """
        if api_key_param:
            return api_key_param
        return os.environ.get(self.token_env_var) if self.token_env_var else None

    def auth_hint(self, api_key):
        """
        Explains how to authenticate a download that was refused.
        """
        if api_key:
            return f"The API key was rejected; check the 'api_key' field or the {self.token_env_var} environment variable."
        return f"The model may require a {self.name} API key: set the {self.token_env_var} environment variable or the 'api_key' field."

    def expected_sha256(self, model_url, response, model_filename, api_key):
        """
        Returns the SHA256 (lowercase hex) that the site publishes for the file being downloaded, or None if unknown.
        response is the download response, after redirects; model_filename is the name of the file it delivers.
        May raise if the site cannot be queried; the download then goes ahead without verification.
        """
        return None


class CivitaiSource(ModelSource):
    name = "Civitai"
    domains = ("civitai.com", "civitai.red", "civitai.green")
    token_env_var = "CIVITAI_TOKEN"

    def expected_sha256(self, model_url, response, model_filename, api_key):
        # Download URLs look like https://civitai.com/api/download/models/<version id>[?fileId=...&type=...&format=...]
        url = urlparse(model_url)
        match = re.match(r"/api/download/models/(\d+)", url.path)
        if not match:
            return None

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        info = requests.get(f"{url.scheme}://{url.netloc}/api/v1/model-versions/{match.group(1)}", headers=headers, timeout=60)
        info.raise_for_status()

        # A version can offer several files (e.g. .safetensors and .pt); find the one being downloaded
        files = info.json().get("files", [])
        file_id = parse_qs(url.query).get("fileId", [None])[0]
        if file_id:
            file = next((f for f in files if str(f.get("id")) == file_id), None)
        else:
            file = next((f for f in files if f.get("name") == model_filename), None)

        sha256 = ((file or {}).get("hashes") or {}).get("SHA256")
        return sha256.lower() if sha256 else None


class HuggingFaceSource(ModelSource):
    name = "HuggingFace"
    domains = ("huggingface.co", "hf.co")
    token_env_var = "HUGGINGFACE_TOKEN"

    def expected_sha256(self, model_url, response, model_filename, api_key):
        # For files stored with Git LFS (which model files are), the resolve URL's redirect carries the
        # file's SHA256 in X-Linked-Etag. Smaller files have a git SHA1 there instead, which is not used.
        for r in (response, *response.history):
            etag = r.headers.get("X-Linked-Etag", "").removeprefix("W/").strip('"').lower()
            if re.fullmatch(r"[0-9a-f]{64}", etag):
                return etag
        return None


MODEL_SOURCES = [CivitaiSource(), HuggingFaceSource()]


def get_source(url):
    """
    Returns the ModelSource that url belongs to, or None for other sites.
    """
    return next((source for source in MODEL_SOURCES if source.matches(url)), None)
