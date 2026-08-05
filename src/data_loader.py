import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(path):
    # load file based on extension
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".txt"):
        df = pd.read_csv(path, sep=';', low_memory=False)
    elif path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format")

    return df


def preprocess_data(df):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # drop first column if it's not numeric (like id/time)
    if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        df = df.iloc[:, 1:]

    # fix decimal format issues (if any)
    df = df.replace(',', '.', regex=True)

    # convert everything to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # remove columns with too many missing values
    df = df.dropna(axis=1, thresh=int(0.7 * len(df)))

    # fill remaining missing values with mean
    df = df.fillna(df.mean())

    # remove columns with no variation
    df = df.loc[:, df.std() != 0]

    # check if any valid features left
    if df.shape[1] == 0:
        raise ValueError("No valid numeric features left after preprocessing!")

    # scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    return scaled_data, scaler


def split_baseline_stream(data, baseline_ratio=0.6):
    # split data into training (baseline) and future (stream)
    split_index = int(len(data) * baseline_ratio)

    baseline_data = data[:split_index]
    stream_data = data[split_index:]

    return baseline_data, stream_data