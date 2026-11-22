from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


API_URL = "https://shadow.caa.edu.cn/api/picture.json"
HOST = "https://shadow.caa.edu.cn"
ROLE_FILTERS = {
    "Sheng": "头茬-生",
    "Dan": "头茬-旦",
    "Jing": "头茬-净",
    "Mo": "头茬-末",
    "Chou": "头茬-丑",
}


def api_page(category: str, page: int, page_size: int = 1000) -> dict[str, Any]:
    params = [
        ("order-by", "sort"),
        ("order-by", "id"),
        ("page", page),
        ("page-size", page_size),
        ("hidden", "false"),
        ("category", category + "*"),
    ]
    response = requests.get(API_URL, params=params, timeout=90)
    response.raise_for_status()
    return response.json()


def collect_manifest() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, category in ROLE_FILTERS.items():
        first = api_page(category, 0)
        total = int(first.get("total_elements", 0))
        total_pages = int(first.get("total_pages", 1))
        pages: dict[int, dict[str, Any]] = {0: first}
        page_numbers = list(range(1, total_pages))

        def fetch(page: int) -> tuple[int, dict[str, Any]]:
            last_error: Exception | None = None
            for attempt in range(5):
                try:
                    return page, api_page(category, page)
                except Exception as exc:
                    last_error = exc
                    time.sleep(1.5 * (attempt + 1))
            assert last_error is not None
            raise last_error

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch, page) for page in page_numbers]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                page, data = future.result()
                pages[page] = data
                if index % 5 == 0 or index == len(futures):
                    print(f"metadata {role}: {index}/{len(futures)} pages", flush=True)

        role_records = []
        for page in range(total_pages):
            role_records.extend(pages[page].get("content", []))
        if len(role_records) != total:
            raise RuntimeError(f"{role}: expected {total} records, got {len(role_records)}")
        missing_url = 0
        for item in role_records:
            record = dict(item)
            record["role"] = role
            if record.get("url"):
                record["source_url"] = HOST + record["url"]
            else:
                record["source_url"] = None
                missing_url += 1
            if record.get("preview_url"):
                record["preview_source_url"] = HOST + record["preview_url"]
            else:
                record["preview_source_url"] = None
            records.append(record)
        print(f"metadata {role}: {len(role_records)} images (missing source URL: {missing_url})", flush=True)
    return records


def extension_for(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix) else ".jpg"


def download_one(record: dict[str, Any], root: Path, preview: bool) -> tuple[str, int, str | None]:
    role = record["role"]
    url = record["preview_source_url"] if preview else record["source_url"]
    output = root / ("preview" if preview else "original") / role / f"{record['id']}{extension_for(url)}"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return "skipped", output.stat().st_size, None

    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "VRAGM-data-retriever/1.0"},
                timeout=(20, 120),
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                raise RuntimeError(f"unexpected content-type {content_type!r}")
            temp = output.with_suffix(output.suffix + ".part")
            temp.write_bytes(response.content)
            if temp.stat().st_size == 0:
                raise RuntimeError("empty response")
            temp.replace(output)
            return "downloaded", output.stat().st_size, None
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    return "failed", 0, repr(last_error)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("皮影数据集") / "shadow_puppetry_dataset")
    parser.add_argument("--preview", action="store_true", help="download museum preview images instead of originals")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    manifest_path = args.root / "metadata.json"
    if manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"loaded existing metadata: {len(records)} records", flush=True)
    else:
        records = collect_manifest()
        manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved metadata: {len(records)} records", flush=True)

    readme = args.root / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Shadow Puppetry image dataset collected from the Digital Museum of Shadow.\n"
            "Source API: https://shadow.caa.edu.cn/api/picture.json\n"
            "Author code: https://github.com/fantao0916/VRAGM\n"
            "Paper: When MLLMs Meet ICH (Information Processing & Management, 2026).\n"
            "The paper reports five role categories: Sheng, Dan, Jing, Mo, Chou.\n"
            "metadata.json retains the original and preview URLs and all museum fields.\n"
            f"Retrieved at: {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )

    target_kind = "preview" if args.preview else "original"
    target_root = args.root / target_kind
    target_root.mkdir(parents=True, exist_ok=True)
    url_key = "preview_source_url" if args.preview else "source_url"
    download_records = [record for record in records if record.get(url_key)]
    missing_records = [record for record in records if not record.get(url_key)]
    if missing_records:
        print(f"skipping {len(missing_records)} records without a downloadable URL", flush=True)
    failed: list[dict[str, Any]] = []
    downloaded = skipped = bytes_done = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(download_one, record, args.root, args.preview) for record in download_records]
        for index, (record, future) in enumerate(zip(download_records, futures), 1):
            status, size, error = future.result()
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed.append({"id": record["id"], "role": record["role"], "url": record["source_url"], "error": error})
            bytes_done += size
            if index % 100 == 0 or index == len(download_records):
                elapsed = max(time.time() - started, 0.1)
                print(
                    f"{target_kind} {index}/{len(download_records)} downloaded={downloaded} skipped={skipped} "
                    f"failed={len(failed)} bytes={bytes_done / 1e9:.2f} GB rate={bytes_done / elapsed / 1e6:.2f} MB/s",
                    flush=True,
                )

    (args.root / f"missing_{target_kind}.json").write_text(
        json.dumps(missing_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.root / f"failed_{target_kind}.json").write_text(
        json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"finished {target_kind}: downloaded={downloaded} skipped={skipped} failed={len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
