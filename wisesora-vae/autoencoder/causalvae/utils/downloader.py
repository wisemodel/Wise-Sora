import gdown
import os

wisesora_cache_home = os.path.expanduser(
    os.getenv("WISESORA_HOME", os.path.join("~/.cache", "wisesora"))
)


def gdown_download(id, fname, cache_dir=None):
    cache_dir = wisesora_cache_home if not cache_dir else cache_dir

    os.makedirs(cache_dir, exist_ok=True)
    destination = os.path.join(cache_dir, fname)
    if os.path.exists(destination):
        return destination

    gdown.download(id=id, output=destination, quiet=False)
    return destination
