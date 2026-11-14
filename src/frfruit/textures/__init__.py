"""Online PBR texture pack downloader, procedural map synthesizer, and cached shader sampler."""

from frfruit.textures.fetcher import (
    DEFAULT_TEXTURE_CACHE,
    SUPPORTED_SOURCES,
    build_material_from_maps,
    download_pbr_texture_pack,
    generate_procedural_pbr_maps,
    search_online_textures,
)
from frfruit.textures.loader import (
    clear_texture_cache,
    compute_triplanar_uvs,
    load_texture_array,
    perturb_normals,
    sample_texture,
)

__all__ = [
    "DEFAULT_TEXTURE_CACHE",
    "SUPPORTED_SOURCES",
    "build_material_from_maps",
    "clear_texture_cache",
    "compute_triplanar_uvs",
    "download_pbr_texture_pack",
    "generate_procedural_pbr_maps",
    "load_texture_array",
    "perturb_normals",
    "sample_texture",
    "search_online_textures",
]
