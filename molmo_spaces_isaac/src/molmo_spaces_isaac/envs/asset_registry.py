"""Registry for discovering and sampling tabletop-suitable THOR USD assets.

Reads ``usd_assets_metadata.json`` (shipped with the package) and locates the
actual USD files under the ``~/.molmospaces`` cache populated by
``ms-download``.  Assets are filtered by bounding-box size so that only objects
that physically fit on a table are offered.

Usage::

    reg = AssetRegistry()                         # auto-discovers cache
    cup_path = reg.sample("Cup")                  # random Cup variant
    paths = reg.sample_n("Apple", 3)              # 3 random Apples
    any_small = reg.sample(max_dim=0.15)          # any object ≤ 15 cm
    all_cats = reg.categories()                   # available categories
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from molmo_spaces_isaac.utils.common import AssetGenMetadata, load_thor_assets_metadata

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_METADATA = _PACKAGE_ROOT / "resources" / "usd_assets_metadata.json"
_DEFAULT_CACHE_DIR = Path.home() / ".molmospaces" / "usd" / "objects" / "thor"

# Maximum bounding-box dimension (metres) for an object to be considered
# "tabletop-suitable".  Objects larger than this are skipped.
_DEFAULT_MAX_DIM = 0.25

# Asset ID pattern: CategoryName_Number  (e.g. "Cup_1", "Apple_30")
_CATEGORY_RE = re.compile(r"^(.+?)_(\d+)$")


def _asset_category(asset_id: str) -> str:
    """Extract the category prefix from an asset ID like ``Cup_1``."""
    m = _CATEGORY_RE.match(asset_id)
    return m.group(1) if m else asset_id


def _find_usd_file(cache_dir: Path, asset_id: str) -> Path | None:
    """Locate the USD file for *asset_id* inside the cache.

    The downloader creates versioned sub-directories (e.g. ``20260128/``).
    Within that, each asset has a ``<id>_mesh/<id>_mesh.usda`` variant that
    the existing ``FrankaPickupEnv`` uses.  Fall back to ``<id>/<id>.usda``
    if the ``_mesh`` variant doesn't exist.
    """
    for version_dir in sorted(cache_dir.iterdir(), reverse=True):
        if not version_dir.is_dir():
            continue
        # Prefer the _mesh variant (used by FrankaPickupEnv)
        mesh_path = version_dir / f"{asset_id}_mesh" / f"{asset_id}_mesh.usda"
        if mesh_path.is_file():
            return mesh_path
        # Fall back to bare asset
        bare_path = version_dir / asset_id / f"{asset_id}.usda"
        if bare_path.is_file():
            return bare_path
    return None


class AssetRegistry:
    """Discovers and samples tabletop-suitable THOR USD assets."""

    def __init__(
        self,
        metadata_path: Path | str = _DEFAULT_METADATA,
        cache_dir: Path | str = _DEFAULT_CACHE_DIR,
        max_dim: float = _DEFAULT_MAX_DIM,
    ) -> None:
        metadata_path = Path(metadata_path)
        self._cache_dir = Path(cache_dir)
        self._max_dim = max_dim

        all_meta = load_thor_assets_metadata(metadata_path)

        # Filter: non-articulated, valid bbox, max dimension within budget,
        # and the USD file actually exists on disk.
        self._assets: dict[str, AssetGenMetadata] = {}
        self._paths: dict[str, Path] = {}
        self._by_category: dict[str, list[str]] = {}

        for asset_id, meta in all_meta.items():
            if meta.articulated:
                continue
            bbox = meta.bbox_size
            if len(bbox) != 3 or any(d <= 0 for d in bbox):
                continue
            if max(bbox) > max_dim:
                continue
            usd = _find_usd_file(self._cache_dir, asset_id)
            if usd is None:
                continue

            self._assets[asset_id] = meta
            self._paths[asset_id] = usd

            cat = _asset_category(asset_id)
            self._by_category.setdefault(cat, []).append(asset_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def categories(self) -> list[str]:
        """Return sorted list of available categories."""
        return sorted(self._by_category)

    def assets_in_category(self, category: str) -> list[str]:
        """Return asset IDs belonging to *category*."""
        return list(self._by_category.get(category, []))

    def usd_path(self, asset_id: str) -> Path:
        """Return the on-disk USD path for *asset_id*."""
        if asset_id not in self._paths:
            raise KeyError(f"Asset '{asset_id}' not found in registry")
        return self._paths[asset_id]

    def bbox(self, asset_id: str) -> tuple[float, float, float]:
        """Return (x, y, z) bounding-box size in metres."""
        m = self._assets[asset_id]
        return (m.bbox_size[0], m.bbox_size[1], m.bbox_size[2])

    def all_asset_ids(self) -> list[str]:
        """Return all registered asset IDs."""
        return list(self._assets)

    def __len__(self) -> int:
        return len(self._assets)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        category: str | None = None,
        *,
        max_dim: float | None = None,
        rng: np.random.Generator | None = None,
        exclude: set[str] | None = None,
    ) -> tuple[str, Path]:
        """Pick one random asset, returning ``(asset_id, usd_path)``.

        Parameters
        ----------
        category:
            If given, restrict to this category.
        max_dim:
            Override the registry-wide max_dim filter for this call.
        rng:
            Numpy random generator (creates one if *None*).
        exclude:
            Asset IDs to skip (useful when sampling without replacement).
        """
        rng = rng or np.random.default_rng()
        exclude = exclude or set()

        if category is not None:
            pool = self._by_category.get(category, [])
        else:
            pool = list(self._assets)

        if max_dim is not None:
            pool = [a for a in pool if max(self._assets[a].bbox_size) <= max_dim]

        pool = [a for a in pool if a not in exclude]

        if not pool:
            raise RuntimeError(
                f"No assets available (category={category!r}, max_dim={max_dim}, "
                f"exclude={len(exclude or [])} items)"
            )

        asset_id = pool[rng.integers(len(pool))]
        return asset_id, self._paths[asset_id]

    def sample_n(
        self,
        n: int,
        category: str | None = None,
        *,
        max_dim: float | None = None,
        rng: np.random.Generator | None = None,
        allow_repeat_category: bool = True,
    ) -> list[tuple[str, Path]]:
        """Sample *n* distinct assets, returning list of ``(asset_id, usd_path)``.

        If *allow_repeat_category* is False, each returned asset will be from
        a different category (useful for varied clutter scenes).
        """
        rng = rng or np.random.default_rng()
        results: list[tuple[str, Path]] = []
        exclude: set[str] = set()
        used_cats: set[str] = set()

        for _ in range(n):
            if category is not None:
                pool = self._by_category.get(category, [])
            else:
                pool = list(self._assets)

            if max_dim is not None:
                pool = [a for a in pool if max(self._assets[a].bbox_size) <= max_dim]
            pool = [a for a in pool if a not in exclude]
            if not allow_repeat_category:
                pool = [a for a in pool if _asset_category(a) not in used_cats]

            if not pool:
                break  # ran out of distinct assets

            asset_id = pool[rng.integers(len(pool))]
            results.append((asset_id, self._paths[asset_id]))
            exclude.add(asset_id)
            used_cats.add(_asset_category(asset_id))

        return results
