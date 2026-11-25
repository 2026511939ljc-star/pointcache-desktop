"""Keep complete alternating frames and selected original records between them."""

import numpy as np


def select(cache, config, times, ids, union, engine):
    engine.require(cache.kind == 'hbjson', 'Key sample mode requires HBJSON input.')
    engine.require(len(times) >= 2, 'Key sample mode requires at least two frames.')
    id_name = config['id_attribute']
    time_name = config['time_attribute']
    position = config['position_attribute']
    velocity = config['velocity_attribute']
    age = config['age_attribute']
    engine.require(engine.attr(cache, position)['size'] == 3, 'Position must have three components.')
    engine.require(len({id_name, time_name, position, velocity, age}) == 5,
                   'ID, time, position, velocity and age attribute names must be distinct.')
    dense = [np.searchsorted(union, frame_ids) for frame_ids in ids]
    last_seen = np.full(len(union), -1, dtype=np.int32)
    for index, frame_ids in enumerate(dense):
        prior = last_seen[frame_ids]
        engine.require(not ((prior >= 0) & (prior != index - 1)).any(),
                       'Key sample mode requires contiguous ID lifetimes in the source.')
        last_seen[frame_ids] = index

    base = set(range(0, len(times), 2)) | {len(times) - 1}
    masks = [np.ones(len(frame.rows), dtype=bool) for frame in cache.frames]
    limits = {position: config['position_tolerance'], velocity: config['velocity_tolerance'],
              age: config['age_tolerance']}
    errors = {attribute['name']: [] for attribute in cache.attrs if attribute['name'] != id_name}
    reasons = {'boundary': 0, 'position_turn': 0}
    for index in range(1, len(times) - 1, 2):
        row_ids = dense[index]
        left_map = np.full(len(union), -1, dtype=np.int32)
        right_map = left_map.copy()
        left_map[dense[index - 1]] = np.arange(len(dense[index - 1]), dtype=np.int32)
        right_map[dense[index + 1]] = np.arange(len(dense[index + 1]), dtype=np.int32)
        left_index, right_index = left_map[row_ids], right_map[row_ids]
        keep = (left_index < 0) | (right_index < 0)
        reasons['boundary'] += int(keep.sum())
        middle = np.flatnonzero(~keep)
        weight = (times[index] - times[index - 1]) / (times[index + 1] - times[index - 1])
        per_attribute = {}
        for attribute in cache.attrs:
            name = attribute['name']
            if name == id_name:
                continue
            lo = engine.values(cache.frames[index - 1].rows[left_index[middle]], attribute)
            raw = engine.values(cache.frames[index].rows[middle], attribute)
            hi = engine.values(cache.frames[index + 1].rows[right_index[middle]], attribute)
            if name in limits or name == time_name:
                engine.require(set(attribute['types']) <= set('fd'),
                               'Interpolated attribute must be floating point: ' + name)
                lo, raw, hi = (values.astype(np.float64) for values in (lo, raw, hi))
                diff = np.linalg.norm(lo + (hi - lo) * weight - raw, axis=1)
                threshold = limits.get(name, config['age_tolerance'])
                bad = diff > threshold
            else:
                bad = ((lo != raw) | (hi != raw)).any(axis=1)
                diff = bad.astype(np.float64)
            keep[middle[bad]] = True
            reasons[name] = reasons.get(name, 0) + int(bad.sum())
            per_attribute[name] = diff
            if name == position and config['protect_turns']:
                delta_in, delta_out = raw - lo, hi - raw
                turn = ((delta_in * delta_out < 0) & (abs(delta_in) > 1e-6)
                        & (abs(delta_out) > 1e-6)).any(axis=1)
                keep[middle[turn]] = True
                reasons['position_turn'] += int(turn.sum())
        masks[index] = keep
        for name, diff in per_attribute.items():
            errors[name].append(diff[~keep[middle]])
    baseline = sum(len(cache.frames[i].rows) for i in base)
    return masks, dict(
        full_frame_numbers=[cache.frames[i].number for i in sorted(base)],
        supplemented_records=sum(int(mask.sum()) for mask in masks) - baseline,
        protection_matches_overlap=reasons,
        attribute_errors={name: engine.statistics(parts) for name, parts in errors.items()},
        samples_without_both_ID_neighbors=0,
    )
