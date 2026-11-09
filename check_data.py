"""Fast image-header/point checks; optional full decoding and pixel-duplicate audit."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

import math
import struct
import sys
from contextlib import contextmanager
from types import SimpleNamespace

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# SAH needs unique float32 point coordinates for its spacing prior.
REJECT_DUPLICATE_POINTS = True


def read_points(path, width, height):
    points, seen = [], set()
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f'{path}:{line_number}: expected exactly two pixel coordinates x y')
        x, y = map(float, fields)
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            raise ValueError(f'{path}:{line_number}: point {(x,y)} outside {width}x{height}')
        if REJECT_DUPLICATE_POINTS:
            # Same precision used by the SAH training loader, without importing torch/numpy.
            point = struct.unpack('ff', struct.pack('ff', x, y))
            if point in seen:
                raise ValueError(f'{path}:{line_number}: Duplicate point coordinates {point}')
            seen.add(point)
        points.append((x, y))
    return points


def image_index(root):
    images = {}
    for path in (root/'JPEGImages').iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in images:
                raise ValueError(f'Ambiguous image ID {path.stem}; multiple extensions')
            images[path.stem] = path
    return images


def read_split(root, split, images):
    manifest = root/'ImageSets'/f'{split}.txt'
    ids = [line.strip() for line in manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'Empty or duplicate IDs in {manifest}')
    samples = []
    for name in ids:
        if '/' in name or chr(92) in name or name in ('.', '..'):
            raise ValueError(f'Expected a filename stem, got {name!r}')
        annotation = root/'PointAnnotations'/f'{name}.txt'
        if name not in images or not annotation.is_file():
            raise FileNotFoundError(f'Missing image or annotation for {name!r}')
        samples.append((name, images[name], annotation))
    return SimpleNamespace(ids=ids, samples=samples)


def data_roots(root):
    paths = [root] if isinstance(root, (str, Path)) else root
    roots = tuple(Path(path).expanduser().resolve() for path in paths)
    if not roots or len(roots) != len(set(roots)):
        raise ValueError('Provide one or more distinct dataset roots')
    return roots


def combine_splits(roots, split, indexes=None):
    """Resolve each manifest against its own dataset, then join sample lists."""
    if indexes is None:
        indexes = [image_index(root) for root in roots]
    samples = []
    sources = [root.name.replace('__', '_') for root in roots]
    if len(set(sources)) != len(sources):
        sources = [f'{i+1}:{name}' for i, name in enumerate(sources)]
    for root, images, source in zip(roots, indexes, sources):
        for name, image, annotation in read_split(root, split, images).samples:
            sample_id = f'{source}__{name}' if len(roots) > 1 else name
            samples.append((sample_id, image, annotation))
    return SimpleNamespace(ids=[name for name, _, _ in samples], samples=samples)


def check_disjoint(left, right, label):
    overlap = set(left.ids) & set(right.ids)
    if overlap:
        raise ValueError(f'{label} IDs overlap: {sorted(overlap)[:3]}')
    images = {path.resolve() for _, path, _ in left.samples}
    for _, path, _ in right.samples:
        if path.resolve() in images:
            raise ValueError(f'{label} share the same image path: {path}')


@contextmanager
def progress_bar(iterable, desc, unit):
    try:
        from tqdm import tqdm
    except ImportError:
        # Optional tqdm: data checks still work in a minimal CPU environment.
        def counted():
            total = len(iterable)
            for i, item in enumerate(iterable, 1):
                yield item
                if i == 1 or i % 100 == 0 or i == total:
                    print(f'{desc}: {i}/{total} {unit}s', file=sys.stderr, flush=True)
        yield counted()
    else:
        with tqdm(iterable, desc=desc, unit=unit, file=sys.stderr,
                  dynamic_ncols=True, mininterval=.5, leave=False) as bar:
            yield bar


def audit(root, check_overlap=False, *, decode=False, show_progress=True):
    roots = data_roots(root)
    reports, image_hashes = {}, {}
    datasets = {}
    indexes = [image_index(path) for path in roots]
    for split in ('train', 'val', 'test'):
        present = [(path/'ImageSets'/f'{split}.txt').is_file() for path in roots]
        if split in ('train', 'val') or all(present):
            if not all(present):
                raise ValueError(f'Expected {split}.txt in each dataset root')
            datasets[split] = combine_splits(roots, split, indexes)
    for left in datasets:
        for right in datasets:
            if left < right:
                check_disjoint(datasets[left], datasets[right], f'{left}/{right}')
    for split, dataset in datasets.items():
        counts, sizes = [], {}
        mode = 'pixels' if check_overlap else 'decode' if decode else 'headers'
        # JSON stays on stdout; progress.py writes transient bars to stderr/TTY.
        if show_progress:
            context = progress_bar(dataset.samples, desc=f'Check {split} ({mode})', unit='image')
        else:
            from contextlib import nullcontext
            context = nullcontext(dataset.samples)
        with context as samples:
            for name, path, annotation in samples:
                try:
                    with Image.open(path) as source:
                        # Image.open is lazy for standard JPEG/PNG: read dimensions
                        # without loading pixels or converting every image to RGB.
                        width, height = source.size
                        points = read_points(annotation, width, height)
                        if decode or check_overlap:
                            source.load()
                        if check_overlap:
                            image = source.convert('RGB')
                            digest = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                            if digest in image_hashes:
                                previous = image_hashes[digest]
                                raise ValueError(f'Duplicate decoded image: {previous} vs {(split,name)}')
                            image_hashes[digest] = (split, name)
                except (OSError, ValueError) as error:
                    raise ValueError(f'{split}/{name}: {error}') from error
                size = f'{width}x{height}'
                sizes[size] = sizes.get(size, 0) + 1
                counts.append(len(points))
        reports[split] = dict(images=len(dataset.samples), points=sum(counts), empty_images=counts.count(0),
                             mean_points=sum(counts)/len(counts), max_points=max(counts), sizes=sizes)
    return reports


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-path', type=Path, nargs='+', required=True)
    p.add_argument('--decode', action='store_true', help='Fully decode images to detect pixel-data corruption; slower')
    p.add_argument('--check-overlap', action='store_true', help='Fully decode and hash pixels to detect duplicates; slower')
    p.add_argument('--no-progress', action='store_true', help='Disable terminal progress bars')
    args = p.parse_args()
    print(json.dumps(audit(args.data_path, args.check_overlap, decode=args.decode,
                           show_progress=not args.no_progress), indent=2))
