import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
from folktables import ACSDataSource, ACSIncome
from folktables.acs import adult_filter
from folktables.load_acs import _STATE_CODES


DATA_DIR = Path(__file__).resolve().parents[1] / "exploration" / "data"
FEATURES = list(ACSIncome.features)
RELATIONSHIP_MAP_2021 = {
    20: 0, 21: 1, 22: 13, 23: 1, 24: 13, 25: 2, 26: 3, 27: 4, 28: 5,
    29: 6, 30: 7, 31: 8, 32: 9, 33: 10, 34: 12, 35: 14, 36: 15,
    37: 16, 38: 17,
}


def harmonize_features(data, year):
    """Return a copy with 2018 and 2021 relationship codes aligned."""
    data = data.copy()
    if year == 2018:
        if data["RELP"].isna().any():
            raise ValueError("Missing 2018 relationship codes")
        data["RELP"] = data["RELP"].replace({11: 15})
    elif year == 2021:
        if data["RELSHIPP"].isna().any():
            raise ValueError("Missing 2021 relationship codes")
        codes = data["RELSHIPP"].dropna().unique()
        if not set(codes).issubset(RELATIONSHIP_MAP_2021):
            raise ValueError("Unexpected 2021 relationship code")
        data["RELP"] = data["RELSHIPP"].map(RELATIONSHIP_MAP_2021)
    else:
        raise ValueError("year must be 2018 or 2021")
    return data


def _load_state_records(year, state):
    """Load features, labels and person IDs in the same filtered row order."""
    source = ACSDataSource(
        survey_year=str(year), horizon="1-Year", survey="person",
        root_dir=str(DATA_DIR),
    )
    data = source.get_data(states=[state], download=True)
    data = harmonize_features(data, int(year))
    filtered = adult_filter(data)
    X, y, _ = ACSIncome.df_to_numpy(data)
    if len(filtered) != len(y):
        raise ValueError("Filtered records and labels are misaligned")
    row_ids = np.array([
        f"{year}:{state}:{serial}:{person}"
        for serial, person in zip(filtered["SERIALNO"], filtered["SPORDER"])
    ])
    return X, y, row_ids


def load_state(year, state):
    """Load one state's ACSIncome features and labels."""
    X, y, _ = _load_state_records(year, state)
    return X, y


def prepare_split(year, states):
    """Combine states and return features, labels, and aligned state IDs."""
    batches = [_load_state_records(year, state) for state in states]
    X = np.concatenate([batch[0] for batch in batches])
    y = np.concatenate([batch[1] for batch in batches])
    state_ids = np.concatenate([
        np.full(len(batch[1]), state) for state, batch in zip(states, batches)
    ])
    row_ids = np.concatenate([batch[2] for batch in batches])
    validate_split(X, y, state_ids, row_ids)
    return X, y, state_ids, row_ids


def validate_split(X, y, state_ids, row_ids=None):
    """Check that split arrays have valid values and matching rows."""
    if X.ndim != 2 or X.shape[1] != len(FEATURES):
        raise ValueError("X must have 10 feature columns")
    if y.ndim != 1 or state_ids.ndim != 1 or len(X) != len(y) or len(y) != len(state_ids):
        raise ValueError("X, y, and state IDs must have aligned rows")
    if not np.isin(y, [0, 1, False, True]).all():
        raise ValueError("y must contain binary labels")
    if not np.isfinite(X).all():
        raise ValueError("X contains missing or non-finite values")
    if row_ids is not None:
        if row_ids.ndim != 1 or len(row_ids) != len(y) or len(np.unique(row_ids)) != len(y):
            raise ValueError("Person IDs must be unique and aligned with rows")
    return True


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_split(X, y, state_ids, row_ids, path, year):
    """Save validated arrays and source/version provenance in an NPZ file."""
    validate_split(X, y, state_ids, row_ids)
    states = np.unique(state_ids).tolist()
    source_files = {}
    for state in states:
        source = DATA_DIR / str(year) / "1-Year" / f"psam_p{_STATE_CODES[state]}.csv"
        source_files[state] = {"name": source.name, "sha256": _sha256(source)}
    metadata = {
        "source": "US Census ACS 1-Year PUMS person survey",
        "year": int(year),
        "folktables_version": importlib.metadata.version("folktables"),
        "python_version": platform.python_version(),
        "features": FEATURES,
        "relationship_harmonization": "RELP/RELSHIPP v1",
        "source_files": source_files,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X=X, y=y, state_ids=state_ids, row_ids=row_ids,
        metadata=json.dumps(metadata),
    )
    return path
