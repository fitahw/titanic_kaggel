import pandas as pd


FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]


class TitanicPreprocessor:
    def __init__(self, seed=67):
        #random seed
        self.seed = seed
        #kolumny featurowe (ważne dla modelu)
        self.feature_cols = None
        #średnia z feature
        self.feature_mean = None
        #deviation z feature
        self.feature_std = None

    def load_and_prepare_data(self, train_path, val_split=0.2):
        data = pd.read_csv(train_path)
        #frac=1 = take all rows, then data.sample = randomly take rows
        data = data.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        #index odcięcia train data od validation data
        split_index = int(len(data) * (1 - val_split))
        train_data = data[:split_index]
        val_data = data[split_index:]

# FITOWANIE modelu na train data
        X_train = self._preprocess(train_data, fit=True)
        y_train = train_data["Survived"].to_numpy()

        # nie fitowanie = zrównanie val_data z train_data (np. w val_data brakuje pasażera z Decku D)
        X_val = self._preprocess(val_data, fit=False)
        y_val = val_data["Survived"].to_numpy()

        return X_train, y_train, X_val, y_val

    def _preprocess(self, data, fit=False):
        if not fit and self.feature_cols is None:
            raise ValueError("Preprocessor must be fitted on training data first.")

        data = data[FEATURE_COLUMNS].copy()

        #Czysczenie age - dla brakujących age ustawiamy na -1, i dodajemy nową tabelę age missing.
        data["Age_missing"] = data["Age"].isna().astype(int)
        data["Age"] = data["Age"].fillna(-1)

        #Czyszczenie kabin - N/A na N lub 0 dla pokoi; Rozbitka "Cabin" na "Deck" i "Room";
        data["Cabin"] = data["Cabin"].fillna("N")
        data["Deck"] = data["Cabin"].str[0]
        data["Room"] = data["Cabin"].str.extract(r"(\d+)", expand=False)
        data["Room"] = data["Room"].fillna("0").astype(int)
        data = data.drop(columns=["Cabin"])

        data["Embarked"] = data["Embarked"].fillna("U")

        # zdummowanie (wszystkie wartości będą numeryczne)
        data = pd.get_dummies(data, columns=["Deck", "Embarked"], dtype=int)
        data = pd.get_dummies(data, columns=["Sex"], drop_first=True, dtype=int)

        #tylko 1 przypadek n.a w Decku T, nie pomoże
        data = data.drop(columns=["Deck_T"], errors="ignore")

# FITOWANIE: feature_cols = wszystkie current cols
        if fit:
            self.feature_cols = list(data.columns)
        else:
            data = data.reindex(columns=self.feature_cols, fill_value=0)

        # floaty bardziej dokładne
        X = data.to_numpy(dtype=float)

        #wyznaczanie średnich / dev.
        if fit:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0)
            self.feature_std[self.feature_std == 0] = 1

        #normalizowanie danych (X - średnia) / dewiacja, zmienia dane w zakres ~-1 - 1
        return (X - self.feature_mean) / self.feature_std
