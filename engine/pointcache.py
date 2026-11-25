"""Inspect or subset SideFX Niagara HBJSON 1.0 without changing retained rows.

Python 3.10+, NumPy. Local files only: no Houdini/Unreal APIs or imports.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import uuid

import numpy as np


class CacheError(ValueError):
    pass


NUMERIC = {
    "b": "i1", "B": "u1", "?": "?", "c": "u1",
    "h": "<i2", "H": "<u2", "l": "<i4", "L": "<u4",
    "q": "<i8", "Q": "<u8", "f": "<f4", "d": "<f8",
}


def check(condition, message):
    if not condition:
        raise CacheError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def exact(stream, size):
    data = stream.read(size)
    check(len(data) == size, f"Truncated file at offset {stream.tell()}: expected {size} bytes.")
    return data


def expect(stream, token):
    position = stream.tell()
    check(exact(stream, len(token)) == token, f"Unsupported/corrupt layout at offset {position}: expected {token!r}.")


def string(stream):
    code = exact(stream, 1)
    formats = {b"B": "<B", b"H": "<H", b"L": "<I"}
    check(code in formats, f"Unsupported string-length marker {code!r}.")
    fmt = formats[code]
    size = struct.unpack(fmt, exact(stream, struct.calcsize(fmt)))[0]
    check(size <= 1024 * 1024, "String exceeds supported 1 MiB limit.")
    try:
        return exact(stream, size).decode("utf-8")
    except UnicodeDecodeError as error:
        raise CacheError("Invalid UTF-8 string.") from error


def key(stream, name):
    actual = string(stream)
    check(actual == name, f"Unsupported layout: expected key {name!r}, got {actual!r}.")


@dataclass
class Header:
    raw: bytes
    offsets: dict
    counts: dict
    attributes: list
    dtype: np.dtype

    def patched(self, **counts):
        result = bytearray(self.raw)
        for name, value in counts.items():
            check(name in self.offsets and 0 <= value < 2**32, f"Invalid uint32 count: {name}={value}.")
            struct.pack_into("<I", result, self.offsets[name], value)
        return bytes(result)

    def attribute(self, name):
        found = [attribute for attribute in self.attributes if attribute["name"] == name]
        check(len(found) == 1, f"Need one explicit {name!r} attribute; do not infer stable IDs from row numbers.")
        return found[0]


def read_header(stream):
    expect(stream, b"{")
    key(stream, "header")
    expect(stream, b"{")
    key(stream, "version")
    check(string(stream) == "1.0", "Only HBJSON version 1.0 is supported.")
    offsets, counts = {}, {}
    for name in ("num_samples", "num_frames", "num_points"):
        key(stream, name)
        offsets[name] = stream.tell()
        counts[name] = struct.unpack("<I", exact(stream, 4))[0]
    key(stream, "num_attrib")
    count = struct.unpack("<H", exact(stream, 2))[0]
    check(count > 0, "No attributes in header.")
    key(stream, "attrib_name")
    expect(stream, b"[")
    names = [string(stream) for _ in range(count)]
    expect(stream, b"]")
    check(len(set(names)) == len(names), "Duplicate attribute names.")
    key(stream, "attrib_size")
    expect(stream, b"[")
    sizes = list(exact(stream, count))
    expect(stream, b"]")
    check(all(size > 0 for size in sizes), "Zero-sized attributes are unsupported.")
    key(stream, "attrib_data_type")
    expect(stream, b"[")
    types = exact(stream, sum(sizes)).decode("ascii", errors="strict")
    expect(stream, b"]")
    check(all(code in NUMERIC for code in types), f"Unsupported component types {set(types) - set(NUMERIC)}.")
    key(stream, "data_type")
    check(string(stream) == "linear", "Only linear fixed-width data is supported.")
    expect(stream, b"}")
    key(stream, "cache_data")
    expect(stream, b"{")
    key(stream, "frames")
    expect(stream, b"[")
    end = stream.tell()
    check(end < 16 * 1024 * 1024, "Header exceeds supported 16 MiB limit.")
    stream.seek(0)
    raw = exact(stream, end)
    fields = [("_begin", "u1")]
    fields.extend((f"c{i}", NUMERIC[code]) for i, code in enumerate(types))
    fields.append(("_end", "u1"))
    attributes, component = [], 0
    for name, size in zip(names, sizes):
        attributes.append({"name": name, "size": size, "types": types[component:component + size],
                           "fields": [f"c{i}" for i in range(component, component + size)]})
        component += size
    return Header(raw, offsets, counts, attributes, np.dtype(fields, align=False))


@dataclass
class Frame:
    prefix: bytes
    count_offset: int
    number: int
    time: float
    rows: np.ndarray

    def patched_prefix(self, count):
        result = bytearray(self.prefix)
        struct.pack_into("<I", result, self.count_offset, count)
        return bytes(result)


def read_frame(stream, header):
    start = stream.tell()
    expect(stream, b"{")
    key(stream, "number")
    number = struct.unpack("<I", exact(stream, 4))[0]
    key(stream, "time")
    time = struct.unpack("<f", exact(stream, 4))[0]
    key(stream, "num_points")
    count_offset = stream.tell() - start
    count = struct.unpack("<I", exact(stream, 4))[0]
    key(stream, "frame_data")
    expect(stream, b"[")
    end = stream.tell()
    stream.seek(start)
    prefix = exact(stream, end - start)
    rows = np.frombuffer(exact(stream, count * header.dtype.itemsize), dtype=header.dtype)
    expect(stream, b"]}")
    return Frame(prefix, count_offset, number, time, rows)


def footer(stream):
    check(stream.read() == b"]}}", "Invalid footer or trailing data.")


def ids_for(rows, header, id_attribute):
    attribute = header.attribute(id_attribute)
    check(attribute["size"] == 1 and attribute["types"] not in ("?", "c"), "ID must be a scalar numeric attribute.")
    values = rows[attribute["fields"][0]]
    if len(values):
        check(np.isfinite(values).all(), "Non-finite ID.")
        check((values >= 0).all() and (values < 2**63).all(), "IDs must fit nonnegative int64.")
    ids = values.astype(np.int64)
    check(np.array_equal(values, ids), "Non-integer ID; refusing to round.")
    check(len(np.unique(ids)) == len(ids), "Duplicate IDs within a frame; repair requires separate evidence.")
    return ids


def finite_rows(rows, header):
    check((rows["_begin"] == ord("[")).all() and (rows["_end"] == ord("]")).all(), "Invalid row delimiters.")
    for name in header.dtype.names:
        if header.dtype[name].kind == "f":
            check(np.isfinite(rows[name]).all(), f"Non-finite numeric attribute component {name}.")


def data_warnings(frames, all_ids, has_id=True):
    warnings = []
    if any(frame.get("reappeared_IDs", 0) for frame in frames):
        warnings.append("IDs reappear after absence; identity reuse or intermittent sampling is not resolved by trimming.")
    if any(frame.get("new_IDs", 0) for frame in frames[1:]):
        warnings.append("Some IDs are born after the first sample; fixed-count consumers must handle pre-birth lookups.")
    if has_id and len(all_ids) and all_ids[-1] >= 2**24:
        warnings.append("IDs reach 2^24; check exact representability and any consumer conversion to float32.")
    if not has_id:
        warnings.append("No explicit stable ID attribute; trajectory trimming is unavailable.")
    return warnings


def scan(source, limits=(), id_attribute="id"):
    """Full read-only inspection. Sources with invalid data are not transformed."""
    source = Path(source).resolve(strict=True)
    check(source.suffix.lower() == ".hbjson", "Expected .hbjson; inspect other formats with their native reader.")
    before = identity(source)
    source_hash = digest(source)
    limits = sorted(set(limits))
    check(all(0 < value < 2**32 for value in limits), "ID limits must be positive uint32 values.")
    all_ids = np.empty(0, dtype=np.int64)
    previous_ids = np.empty(0, dtype=np.int64)
    selected_counts = {limit: [] for limit in limits}
    frames, total = [], 0
    has_id = True
    with source.open("rb") as stream:
        header = read_header(stream)
        has_id = any(attribute["name"] == id_attribute for attribute in header.attributes)
        check(has_id or not limits, f"No {id_attribute!r} attribute; cannot estimate ID subsets.")
        check(header.counts["num_frames"] > 0, "Empty frame array.")
        for index in range(header.counts["num_frames"]):
            frame = read_frame(stream, header)
            finite_rows(frame.rows, header)
            check(math.isfinite(frame.time), "Non-finite frame time.")
            check(not frames or frame.time > frames[-1]["time_seconds"], "Frame times must be strictly increasing; do not guess FPS.")
            info = {"index": index, "frame_number": frame.number, "time_seconds": frame.time, "samples": len(frame.rows)}
            if has_id:
                ids = ids_for(frame.rows, header, id_attribute)
                unique = np.sort(ids)
                new = np.setdiff1d(unique, all_ids, assume_unique=True)
                appeared = np.setdiff1d(unique, previous_ids, assume_unique=True)
                info.update({
                    "id_min": int(unique[0]) if len(unique) else None,
                    "id_max": int(unique[-1]) if len(unique) else None,
                    "new_IDs": len(new), "disappeared_IDs": len(np.setdiff1d(previous_ids, unique, assume_unique=True)),
                    "reappeared_IDs": len(appeared) - len(new),
                    "IDs_equal_row_numbers": bool(np.array_equal(ids, np.arange(len(ids)))),
                })
                for limit in limits:
                    selected_counts[limit].append(int(np.count_nonzero(ids < limit)))
                all_ids = np.union1d(all_ids, unique)
                previous_ids = unique
            frames.append(info)
            total += len(frame.rows)
        footer(stream)
    check(total == header.counts["num_samples"], "Header num_samples differs from observed samples.")
    if has_id:
        check(len(all_ids) == header.counts["num_points"], "Header num_points differs from observed distinct IDs; inspect separately.")
    check(identity(source) == before and digest(source) == source_hash, "Source changed during inspection.")
    warnings = data_warnings(frames, all_ids, has_id)
    estimates = []
    overhead = before[0] - total * header.dtype.itemsize
    for limit in limits:
        selected_ids = all_ids[all_ids < limit]
        samples = sum(selected_counts[limit])
        size = overhead + samples * header.dtype.itemsize
        estimates.append({
            "id_exclusive_limit": limit, "unique_IDs": len(selected_ids), "samples": samples,
            "bytes": size, "MiB": size / 1024**2,
            "reduction_percent": (1 - size / before[0]) * 100,
            "all_requested_IDs_present_somewhere": len(selected_ids) == limit,
            "per_frame_samples": selected_counts[limit],
        })
    return {
        "source": str(source), "source_sha256": source_hash, "source_bytes": before[0],
        "source_MiB": before[0] / 1024**2, "schema": "SideFX Niagara HBJSON 1.0 linear numeric",
        "header": header.counts, "attributes": header.attributes, "record_bytes": header.dtype.itemsize,
        "id_attribute": id_attribute if has_id else None,
        "observed_unique_IDs": len(all_ids) if has_id else None,
        "time_range_seconds": [frames[0]["time_seconds"], frames[-1]["time_seconds"]],
        "frame_count": len(frames), "frames": frames, "subset_estimates": estimates,
        "warnings": warnings, "source_unchanged": True, "validation_passed": True,
    }


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def verify_subset(source, target, limit, id_attribute, expected_header, expected_samples):
    per_frame = []
    unique = np.empty(0, dtype=np.int64)
    previous_ids = unique
    with source.open("rb") as original, target.open("rb") as output:
        before, after = read_header(original), read_header(output)
        check(after.raw == expected_header, "Output header changed unexpectedly.")
        count = 0
        for index in range(before.counts["num_frames"]):
            a, b = read_frame(original, before), read_frame(output, after)
            ids = ids_for(a.rows, before, id_attribute)
            expected = a.rows[ids < limit]
            check(b.prefix == a.patched_prefix(len(expected)), f"Frame metadata changed at {index}.")
            check(b.rows.tobytes() == expected.tobytes(), f"Retained row bytes differ at frame {index}.")
            finite_rows(b.rows, after)
            kept_ids = ids_for(b.rows, after, id_attribute)
            check((kept_ids < limit).all(), "Output has an out-of-range ID.")
            new = np.setdiff1d(kept_ids, unique, assume_unique=True)
            appeared = np.setdiff1d(kept_ids, previous_ids, assume_unique=True)
            unique = np.union1d(unique, kept_ids)
            per_frame.append({"frame_number": b.number, "time_seconds": b.time, "samples": len(b.rows),
                              "new_IDs": len(new), "reappeared_IDs": len(appeared) - len(new),
                              "payload_sha256": hashlib.sha256(b.rows.tobytes()).hexdigest(),
                              "source_filtered_payload_byte_identical": True})
            previous_ids = np.sort(kept_ids)
            count += len(b.rows)
        footer(original)
        footer(output)
    check(count == expected_samples == after.counts["num_samples"], "Output sample count mismatch.")
    check(len(unique) == after.counts["num_points"], "Output distinct ID count mismatch.")
    return per_frame, data_warnings(per_frame, unique)


def trim(source, output, limit, id_attribute="id"):
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    check(output.suffix.lower() == ".hbjson", "Output must end in .hbjson.")
    check(source != output, "Output must differ from source.")
    report_path = output.with_suffix(".validation.json")
    notes_path = output.with_suffix(".README.txt")
    for path in (output, report_path, notes_path):
        check(not path.exists(), f"Refusing to overwrite {path}.")
    plan = scan(source, [limit], id_attribute)
    before = identity(source)
    estimate = plan["subset_estimates"][0]
    check(estimate["unique_IDs"] > 0, "Selected ID range is empty.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + "." + uuid.uuid4().hex + ".part")
    # The source is never opened for writing. Staging is unique; failures retain
    # it for diagnosis rather than publishing an unchecked result.
    with source.open("rb") as stream, staging.open("xb") as destination:
        header = read_header(stream)
        expected_header = header.patched(num_points=estimate["unique_IDs"], num_samples=estimate["samples"])
        destination.write(expected_header)
        for _ in range(header.counts["num_frames"]):
            frame = read_frame(stream, header)
            ids = ids_for(frame.rows, header, id_attribute)
            selected = frame.rows[ids < limit]
            destination.write(frame.patched_prefix(len(selected)))
            destination.write(selected.tobytes())
            destination.write(b"]}")
        footer(stream)
        destination.write(b"]}}")
        destination.flush()
        os.fsync(destination.fileno())
    check(staging.stat().st_size == estimate["bytes"], "Estimated and actual output size differ.")
    frames, warnings = verify_subset(source, staging, limit, id_attribute, expected_header, estimate["samples"])
    check(identity(source) == before and digest(source) == plan["source_sha256"], "Source changed during processing.")
    output_hash = digest(staging)
    # Windows rename refuses existing destinations. On POSIX, link publishes the
    # complete validated inode exclusively without rename's overwrite behavior.
    if os.name == "nt":
        staging.rename(output)
    else:
        os.link(staging, output)
        staging.unlink()
    check(digest(output) == output_hash, "Published file differs from validated staging file.")
    report = {
        "validation_passed": True, "source": str(source), "source_sha256": plan["source_sha256"],
        "source_bytes": plan["source_bytes"], "source_preserved_as_rollback": True,
        "source_unchanged": True, "output": str(output), "output_sha256": output_hash,
        "output_bytes": output.stat().st_size, "output_MiB": estimate["MiB"],
        "reduction_percent": estimate["reduction_percent"], "id_attribute": id_attribute,
        "retained_raw_ID_range": [0, limit - 1], "observed_unique_IDs": estimate["unique_IDs"],
        "all_requested_IDs_present_somewhere": estimate["all_requested_IDs_present_somewhere"],
        "samples": estimate["samples"], "frame_count": plan["frame_count"],
        "time_range_seconds": plan["time_range_seconds"],
        "retained_all_attribute_bytes_and_order_unchanged": True,
        "frame_numbers_and_times_unchanged": True, "only_counts_rewritten": True,
        "source_warnings": plan["warnings"], "warnings": warnings, "frames": frames,
        "UE_imported": False, "GPU_compiled": False, "save_reload_checked": False,
        "runtime_or_visual_validation": "not performed", "Niagara_or_project_assets_written": False,
    }
    write_json_new(report_path, report)
    with notes_path.open("x", encoding="utf-8-sig") as notes:
        notes.write(
            "Houdini 点云缓存 ID 范围裁剪\n\n"
            f"源文件保留：{source}\n输出：{output}\n"
            f"保留原始 {id_attribute} 范围：0–{limit - 1}；实际独立 ID：{estimate['unique_IDs']}\n"
            f"体积：{plan['source_MiB']:.3f} → {estimate['MiB']:.3f} MiB；减少 {estimate['reduction_percent']:.3f}%\n"
            f"样本帧：{plan['frame_count']}；时间范围：{plan['time_range_seconds']} 秒（未更改）\n"
            "保留记录的所有属性和顺序逐字节一致；只删范围外记录并更新计数。\n"
            "未补齐出生记录、未重编号、未抽帧、未修改位置/速度/生命周期。\n"
            "原文件完整保留，处理前后 SHA256 一致，作为回退。\n"
            "本输出未导入 UE，未进行 GPU 编译、保存重载或视觉验收。\n"
            "未修改 Niagara、美术调参、引用或渲染设置。\n"
            "未来需要范围外的 ID 时须重新生成。文件缩小不代表显存/帧率同比改善。\n"
            f"输出 SHA256：{output_hash}\n详细验证：{report_path}\n"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspection = commands.add_parser("inspect", help="Read-only schema, identity, timeline and exact subset-size analysis.")
    inspection.add_argument("source", type=Path)
    inspection.add_argument("--limits", type=int, nargs="*", default=[])
    inspection.add_argument("--id-attribute", default="id")
    inspection.add_argument("--report", type=Path)
    subset = commands.add_parser("trim", help="Keep raw ID < N; preserve every time and retained byte.")
    subset.add_argument("source", type=Path)
    subset.add_argument("--max-points", type=int, required=True)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--id-attribute", default="id")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            if args.report:
                check(args.report.resolve() != args.source.resolve(), "Report cannot overwrite source.")
                check(not args.report.exists(), "Report exists; choose a new report path.")
            result = scan(args.source, args.limits, args.id_attribute)
            if args.report:
                write_json_new(args.report, result)
        else:
            result = trim(args.source, args.output, args.max_points, args.id_attribute)
        compact = {name: value for name, value in result.items() if name != "frames"}
        print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (CacheError, OSError, UnicodeError, struct.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
