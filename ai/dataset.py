import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Loader(Dataset):

    _raw: pd.DataFrame = pd.DataFrame()
    _data: pd.DataFrame = pd.DataFrame()

    # OneHotEncoder instance for encoding "day_of_week" column
    _dof_ohe: OneHotEncoder = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore"
    )

    # StandardScaler instance for scaling "temperature" column
    _t_sc = StandardScaler()

    # StandardScaler instance for scaling "rain" column
    _r_sc = StandardScaler()

    # StandardScaler instance for scaling "week" column
    _w_sc = StandardScaler()

    # StandardScaler instance for scaling "meetings_reservations" column
    _me_sc = StandardScaler()

    # List of feature column names and target column name
    _feature_cols: list[str] = []
    _target_col: str = "GLOBAL"

    def __init__(self, path: str):
        # Read the raw data from the CSV file
        self._raw = pd.read_csv(path, sep=";")

        # Map of original column names to new names
        column_mapping = {
            "jour_feriÃ.": "public_holiday",
            "pont.congÃ.": "bridge_day",
            "holiday": "holiday",
            "jour_semaine": "day_of_week",
            # "Semaine": "week",
            # "Temp": "temperature",
            # "pluie": "rain",
            "autre": "other",
            # "Greve_nationale": "national_strike",
            # "SNCF": "train_strike",
            # "prof_nationale": "education_strike",
            "Total_reservations": "meetings_reservations",
            self._target_col: self._target_col,
        }

        # Select columns using original names
        original_feature_cols = [
            "jour_feriÃ.",
            "pont.congÃ.",
            "holiday",
            "jour_semaine",
            # "Semaine",
            # "Temp",
            # "pluie",
            "autre",
            # "Greve_nationale",
            # "SNCF",
            # "prof_nationale",
            "Total_reservations",
        ]

        # Clean the data by selecting only the relevant columns
        df = self._raw[original_feature_cols + [self._target_col]]

        # Rename columns for better readability
        df = df.rename(columns=column_mapping)

        # Update the feature columns list with the new column names
        self._feature_cols = [x for x in df.columns if x != self._target_col]

        # Store the cleaned data in the _data attribute
        self._data = df

    def encode(self) -> Loader:
        df = self._data.copy()

        # One hot encode the 'day_of_week' column using sklearn
        ohe = self._dof_ohe.fit_transform(df[["day_of_week"]])

        # Create a new DataFrame from the one hot encoded data
        ohe_df = pd.DataFrame(
            ohe, columns=self._dof_ohe.get_feature_names_out(["day_of_week"])
        )

        # Drop the original 'day_of_week' column
        df = df.drop(columns=["day_of_week"])

        # Concatenate the original DataFrame with the one hot encoded DataFrame
        df = pd.concat([df, ohe_df], axis=1)

        # Normalize the 'temperature' column
        # df["temperature"] = self._t_sc.fit_transform(df[["temperature"]])

        # Normalize the 'rain' column
        # df["rain"] = self._r_sc.fit_transform(df[["rain"]])

        # Normalize the 'week' column
        # df["week"] = self._w_sc.fit_transform(df[["week"]])

        # Normalize the 'meetings_reservations' column
        df["meetings_reservations"] = self._me_sc.fit_transform(
            df[["meetings_reservations"]]
        )

        # Update the feature columns list with the new column names
        self._feature_cols = [x for x in df.columns if x != self._target_col]

        print(len(self._feature_cols))

        # Store the encoded data in the _data attribute
        self._data = df

        return self

    def split(self, sizes: tuple[float, float, float]) -> tuple[Subset, Subset, Subset]:
        """Split the dataset into train, validation, and test sets based on the provided sizes."""

        # Calculate the number of samples for each split
        total_size = len(self._data)
        train_size = int(sizes[0] * total_size)
        val_size = int(sizes[1] * total_size)

        # Create Subset objects for train, validation, and test sets
        train_dataset = Subset(self, range(train_size))
        val_dataset = Subset(self, range(train_size, train_size + val_size))
        test_dataset = Subset(self, range(train_size + val_size, total_size))

        return train_dataset, val_dataset, test_dataset

    def get_loaders(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        sizes: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders for train, validation, and test sets."""
        train_dataset, val_dataset, test_dataset = self.split(sizes)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader, val_loader, test_loader

    def get_features_and_targets(self):
        """Return the features and targets as separate DataFrames."""
        features = self._data[self._feature_cols]
        targets = self._data[self._target_col]

        return features, targets

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return (
            self._data.iloc[index][self._feature_cols].values,
            self._data.iloc[index][self._target_col],
        )
