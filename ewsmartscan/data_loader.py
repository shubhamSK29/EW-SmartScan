"""Load synthetic time-stamped pulse descriptor words from HDF5."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "config_0.h5"


@dataclass(frozen=True)
class ReceiverConfig:
    """Receiver and scan parameters needed by the replay environment."""

    dwell_centres_mhz: np.ndarray
    dwell_times_s: np.ndarray
    freq_range_mhz: tuple[float, float]
    bandwidth_mhz: float
    sensitivity_dbm: float
    scan_mode: str
    collection_time_s: float


@dataclass(frozen=True)
class TsrdDataset:
    """Pulse data and metadata loaded from one TSRD collection."""

    pdws: pd.DataFrame
    receiver: ReceiverConfig
    emitters: dict[int, str]
    feature_names: list[str]
    metadata: dict[str, object]


def _decode(value: Any) -> Any:
    """Decode HDF5 byte strings while preserving numeric values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return [_decode(item) for item in value.tolist()]
        if value.dtype.kind == "O":
            return [_decode(item) for item in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return _decode(value.item())
    return value


def _attrs(group: h5py.Group) -> dict[str, object]:
    return {str(key): _decode(value) for key, value in group.attrs.items()}


def _dataset_values(group: h5py.Group, name: str) -> np.ndarray:
    return np.asarray(group[name][()])


def load_tsrd(path: str | Path = DEFAULT_DATA_PATH) -> TsrdDataset:
    """Load a TSRD HDF5 file into typed tabular data and scan metadata."""
    with h5py.File(Path(path), "r") as handle:
        metadata_group = handle["metadata"]
        receiver_group = metadata_group["receiver"]
        raw_data = np.asarray(handle["data"][()])
        raw_labels = np.asarray(handle["labels"][()]).reshape(-1)
        feature_names = [str(item) for item in _decode(metadata_group["feature_names"][()])]

        columns = {
            "ToA": "toa_us",
            "Frequency": "frequency_mhz",
            "PulseWidth": "pulse_width_us",
            "AoA": "aoa_deg",
            "Amplitude": "amplitude_dbm",
        }
        frame = pd.DataFrame(
            {
                columns.get(name, name): raw_data[:, index]
                for index, name in enumerate(feature_names)
            }
        )
        frame["toa_s"] = frame["toa_us"] / 1e6
        frame["emitter_id"] = raw_labels.astype(int)
        ordered_columns = [
            "toa_us",
            "toa_s",
            "frequency_mhz",
            "pulse_width_us",
            "aoa_deg",
            "amplitude_dbm",
            "emitter_id",
        ]
        frame = frame[ordered_columns]
        if not frame["toa_us"].is_monotonic_increasing:
            frame = frame.sort_values("toa_us", kind="stable").reset_index(drop=True)

        receiver_attrs = _attrs(receiver_group)
        collection_attrs = _attrs(metadata_group)
        bandwidth = receiver_attrs.get(
            "bandwidth_mhz", receiver_attrs.get("bandwith_mhz", 0.0)
        )
        receiver = ReceiverConfig(
            dwell_centres_mhz=_dataset_values(
                receiver_group, "dwell_centres_mhz"
            ).astype(float),
            dwell_times_s=_dataset_values(receiver_group, "dwell_times_s").astype(float),
            freq_range_mhz=tuple(
                float(item)
                for item in _dataset_values(receiver_group, "freq_range_mhz").reshape(-1)
            ),
            bandwidth_mhz=float(bandwidth),
            sensitivity_dbm=float(receiver_attrs.get("sensitivity_dbm", -np.inf)),
            scan_mode=str(receiver_attrs.get("scan_mode", "")),
            collection_time_s=float(collection_attrs.get("collection_time_s", 0.0)),
        )

        emitters: dict[int, str] = {}
        transmitter_group = metadata_group["transmitters"]
        for name, group in transmitter_group.items():
            if not isinstance(group, h5py.Group):
                continue
            suffix = name.rsplit("_", 1)[-1]
            if suffix.isdigit():
                emitters[int(suffix)] = str(_decode(group.attrs.get("function", "")))

        metadata = collection_attrs
        metadata["receiver"] = receiver_attrs

    return TsrdDataset(
        pdws=frame,
        receiver=receiver,
        emitters=emitters,
        feature_names=feature_names,
        metadata=metadata,
    )
