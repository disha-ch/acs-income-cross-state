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
    """Build features and labels from one filtered dataframe in row order."""
    source = ACSDataSource(
        survey_year=str(year), horizon="1-Year", survey="person",
        root_dir=str(DATA_DIR),
    )
    data = source.get_data(states=[state], download=True)
    data = harmonize_features(data, int(year))
    filtered = adult_filter(data)
    X = filtered[FEATURES].to_numpy()
    X = np.nan_to_num(X)
    y = ACSIncome.target_transform(filtered[ACSIncome.target]).to_numpy()
    return X, y


def load_state(year, state):
    """Load one state's ACSIncome features and labels."""
    return _load_state_records(year, state)


def prepare_split(year, states):
    """Combine states and return features, labels, and aligned state IDs."""
    batches = [_load_state_records(year, state) for state in states]
    X = np.concatenate([batch[0] for batch in batches])
    y = np.concatenate([batch[1] for batch in batches])
    state_ids = np.concatenate([
        np.full(len(batch[1]), state) for state, batch in zip(states, batches)
    ])
    validate_split(X, y, state_ids)
    return X, y, state_ids


def validate_split(X, y, state_ids):
    """Check that split arrays have valid values and matching rows."""
    if X.ndim != 2 or X.shape[1] != len(FEATURES):
        raise ValueError("X must have 10 feature columns")
    if y.ndim != 1 or state_ids.ndim != 1 or len(X) != len(y) or len(y) != len(state_ids):
        raise ValueError("X, y, and state IDs must have aligned rows")
    if not np.isin(y, [0, 1, False, True]).all():
        raise ValueError("y must contain binary labels")
    if not np.isfinite(X).all():
        raise ValueError("X contains missing or non-finite values")
    return True


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_split(X, y, state_ids, path, year):
    """Save validated arrays and source/version provenance in an NPZ file."""
    validate_split(X, y, state_ids)
    states = np.unique(state_ids).tolist()
    source_files = {}
    for state in states:
        source = DATA_DIR / str(year) / "1-Year" / f"psam_p{_STATE_CODES[state]}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"ACS source file not found: {source}")
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
        path, X=X, y=y, state_ids=state_ids,
        feature_names=np.asarray(FEATURES), metadata=json.dumps(metadata),
    )
    return path


def main():
    """Prepare the provisional train, validation, and hidden partitions."""
    root = Path(__file__).resolve().parents[1]
    agent_data = root / "environment" / "task_inputs"
    hidden_data = root / "tests" / "hidden_data"
    partitions = [
        ("train", 2018, ["CA", "TX", "NY"], agent_data / "train.npz"),
        ("val", 2018, ["FL"], agent_data / "val.npz"),
        ("hidden_2018", 2018, ["MS", "WV", "NM", "AR", "LA", "MT"], hidden_data / "hidden_2018.npz"),
        ("hidden_2021", 2021, ["MS", "WV", "NM", "AR", "LA", "MT"], hidden_data / "hidden_2021.npz"),
    ]
    train_states = set(partitions[0][2])
    validation_states = set(partitions[1][2])
    hidden_states = set(partitions[2][2])
    if (train_states & validation_states or train_states & hidden_states
            or validation_states & hidden_states):
        raise ValueError("Training, validation, and hidden states must be disjoint")

    for name, year, states, path in partitions:
        X, y, state_ids = prepare_split(year, states)
        validate_split(X, y, state_ids)
        save_split(X, y, state_ids, path, year)
        with np.load(path, allow_pickle=False) as archive:
            if not (np.array_equal(archive["X"], X)
                    and np.array_equal(archive["y"], y)
                    and np.array_equal(archive["state_ids"], state_ids)):
                raise ValueError(f"Saved arrays failed round-trip validation: {path}")
        print(
            f"{name}: {len(y):,} rows, {X.shape[1]} features, "
            f"positive rate {y.mean():.4f}; saved {path}"
        )

    metadata = {
        "features": FEATURES,
        "label": "PINCP > 50000",
        "filter": ["AGEP > 16", "PINCP > 100", "WKHP > 0", "PWGTP >= 1"],
        "relationship_harmonization": "RELP/RELSHIPP v1",
    }
    agent_data.mkdir(parents=True, exist_ok=True)
    (agent_data / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
