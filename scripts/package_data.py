from pathlib import Path

from folktables import ACSDataSource, ACSIncome


DATA_DIR = Path(__file__).resolve().parents[1] / "exploration" / "data"


def load_state(year, state):
    """Load one state's ACSIncome features and labels."""
    source = ACSDataSource(
        survey_year=str(year), horizon="1-Year", survey="person",
        root_dir=str(DATA_DIR),
    )
    data = source.get_data(states=[state], download=True)
    X, y, _ = ACSIncome.df_to_numpy(data)
    return X, y
