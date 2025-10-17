import math
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from torch.utils.data import DataLoader, Dataset

REGION_DEFINITIONS: Dict[str, Sequence[str]] = {
    "Beograd i okolina": [
        "BEOGRAD",
        "Beograd",
        "Altina",
        "Banovo Brdo",
        "Barajevo",
        "Barič",
        "Batajnica",
        "Boleč",
        "Bežanijska Kosa",
        "Borča",
        "Braće Jerković",
        "Cerak",
        "Čukarica",
        "Dobanovci",
        "Dorćol",
        "Grocka",
        "Jajinci",
        "Kanarevo Brdo",
        "Karaburma",
        "Kaluđerica",
        "Krnjača",
        "Lazarevac",
        "Mali Mokri Lug",
        "Mirijevo",
        "Mladenovac",
        "Novi Beograd",
        "Palilula",
        "Resnik",
        "Rakovica",
        "Ralja",
        "Senjak",
        "Sopot",

        "Surčin",
        "Ugrinovci",
        "Voždovac",
        "Vranić",
        "Vračar",
        "Vrčin",
        "ZEMUN",
        "Zemun",
        "Zemun Polje",
        "Zvezdara",
        "Žarkovo",
        "Železnik",
    ],
    "Vojvodina": [
        "Ada",
        "Alibunar",
        "Apatin",
        "Aradac",
        "Bajmok",
        "Bač",
        "Bačka Palanka",
        "Bačka Topola",
        "Bački Jarak",
        "Bački Petrovac",
        "Bačko Gradište",
        "Bela Crkva",
        "Bečej",
        "Beška",
        "Bezdan",
        "Crvenka",
        "Deliblato",
        "Deronje",
        "Dobrinci",
        "Dolovo",
        "Ečka",
        "Futog",
        "Golubinci",
        "Hetin",
        "Inđija",
        "Izbište",
        "Karađorđevo",
        "Kanjiža",
        "Kać",
        "Kačarevo",
        "Kikinda",
        "Klenak",
        "Kovačica",
        "Kovin",
        "Krajišnik",
        "Kula",
        "Kumane",
        "Kuzmin",
        "Laćarak",
        "Lazarevo",
        "Ljutovo",
        "Lovćenac",
        "Mala Bosna",
        "Mačvanska Mitrovica",
        "Mokrin",
        "Mol",
        "Novi Banovci",
        "Nova Pazova",
        "Novi Sad",
        "Novi Bečej",
        "Obrovac",
        "Omoljica",
        "Pančevo",
        "Petrovaradin",
        "Prigrevica",
        "Radičević",
        "Ravni Topolovac",
        "Ruma",
        "Sakule",
        "Sečanj",
        "Srbobran",
        "Sremska Kamenica",
        "Sremska Mitrovica",
        "Sremski Karlovci",
        "Stajićevo",
        "Stanišić",
        "Stara Pazova",
        "Stari Ledinci",
        "Subotica",
        "Svetozar Miletić",
        "Temerin",
        "Torak",
        "Veternik",
        "Vrbas",
        "Vrdnik",
        "Vršac",
        "Zmajevo",
        "Zrenjanin",
        "Čenta",
        "Čurug",
        "Šajkaš",
        "Šid",
        "Šimanovci",
        "Žabalj",
    ],
    "Zapadna Srbija": [
        "Arilje",
        "Bajina Bašta",
        "Banovo Polje",
        "Bogatić",
        "Bogosavac",
        "Brestovac",
        "GORNJI MILANOVAC",
        "Gornji Milanovac",
        "Ivanjica",
        "Koceljeva",
        "Kosjerić",
        "Kraljevo",
        "Krupanj",
        "Kusadak",
        "Lajkovac (Varoš)",
        "Ljig",
        "Loznica",
        "Lučani (Selo)",
        "Majur",
        "Mionica (Varošica)",
        "Mrčajevci",
        "Mali Zvornik",
        "Nova Varoš",
        "Novi Pazar",
        "Obrenovac",
        "Osečina (Varošica)",
        "Požega",
        "Priboj",
        "Prijepolje",
        "Sevojno",
        "Sjenica",
        "Šabac",
        "Ševarice",
        "Tutin",
        "Ub",
        "Užice",
        "Valjevo",
        "Varvarin",
        "Zlatibor",
        "Čačak",
    ],
    "Istočna Srbija": [
        "Bela Palanka",
        "Bor",
        "Despotovac",
        "Kladovo",
        "Knjaževac",
        "Kostolac",
        "Kučevo",
        "Lipnički Šor",
        "Majdanpek",
        "Negotin",
        "Paraćin",
        "Petrovac Na Mlavi",
        "Požarevac",
        "Ražanj",
        "Rekovac",
        "Smederevo",
        "Svilajnac",
        "Velika Plana",
        "Veliko Gradište",
        "Zaječar",
        "Ćuprija",
    ],
    "Južna Srbija": [
        "Aleksandrovac",
        "Aleksinac",
        "Aranđelovac",
        "Babušnica",
        "Batočina",
        "Blace",
        "Bojnik",
        "Bratinac",
        "Brus",
        "Darosava",
        "Deveti Maj",
        "Jagodina",
        "Jasenovik",
        "Jevremovac",
        "Jošanička Banja",
        "Knić",
        "Kragujevac",
        "KRAGUJEVAC",
        "Kruševac",
        "Kuršumlija",
        "Lapovo (Varošica)",
        "Lebane",
        "Leskovac",
        "Malošište",
        "Markovac",
        "Masloševo",
        "Mramor",
        "Niš",
        "NIŠ",
        "Obrež",
        "Pirot",
        "Poljna",
        "Prokuplje",
        "Predejane (Varoš)",
        "Preševo",
        "Ribari",
        "Sokobanja",
        "Surdulica",
        "Tibužde",
        "Topola",
        "Trstenik",
        "Vlasotince",
        "Vladičin Han",
        "Vranje",
        "Vrnjačka Banja",
    ],
}


ALPHA_BY_BRAND: Dict[str, float] = {
    "BMW": 1.35,
    "Audi": 1.3,
    "Mercedes-Benz": 1.4,
    "Volkswagen": 1.25,
    "Toyota": 1.2,
    "Peugeot": 1.15,
    "Renault": 1.1,
    "Fiat": 1.05,
    "Opel": 1.1,
    "Škoda": 1.18,
    "Ford": 1.15,
    "Hyundai": 1.15,
    "Kia": 1.12,
    "Seat": 1.13,
    "Volvo": 1.28,
}


class SpatialTemporalSplitter:
    def __init__(
        self,
        regions: Sequence[str],
        region_column: str,
        year_column: str,
        recent_year_threshold: int = 2018,
    ):
        self.regions = regions
        self.region_column = region_column
        self.year_column = year_column
        self.recent_year_threshold = recent_year_threshold

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[Sequence[int]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for region in self.regions:
            region_mask = X[self.region_column] == region
            recent_mask = (X[self.year_column] > self.recent_year_threshold) & (~region_mask)
            val_indices = np.where(region_mask | recent_mask)[0]
            train_indices = np.setdiff1d(np.arange(len(X)), val_indices)
            yield train_indices, val_indices

    def get_n_splits(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None, groups: Optional[Sequence[int]] = None) -> int:
        return len(self.regions)


class TabularDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: Sequence[str],
        num_cols: Sequence[str],
        target: pd.Series,
        cat_mappings: Dict[str, Dict[str, int]],
    ):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.cat_mappings = cat_mappings
        self.target = target.values.astype(np.float32)

        cat_arrays = []
        for col in self.cat_cols:
            mapping = self.cat_mappings[col]
            unknown_idx = mapping["__unknown__"]
            cat_arrays.append(df[col].map(mapping).fillna(unknown_idx).astype(int).values)
        self.cats = np.stack(cat_arrays, axis=1)

        self.nums = df[self.num_cols].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cats = torch.tensor(self.cats[idx], dtype=torch.long)
        nums = torch.tensor(self.nums[idx], dtype=torch.float32)
        y = torch.tensor(self.target[idx], dtype=torch.float32)
        return cats, nums, y


class TabTransformer(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        num_numeric: int,
        emb_dim: int = 32,
        depth: int = 4,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, emb_dim, padding_idx=cardinality)
            for cardinality in cardinalities
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            dim_feedforward=emb_dim * 4,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.num_numeric = num_numeric
        if num_numeric > 0:
            self.numeric_projection = nn.Sequential(
                nn.Linear(num_numeric, emb_dim),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
            )
        else:
            self.numeric_projection = None
        input_dim = emb_dim * 2 if num_numeric > 0 else emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_cats: torch.Tensor, x_nums: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            embeddings.append(emb(x_cats[:, i]))
        x = torch.stack(embeddings, dim=1)
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.transformer(x)
        cls_rep = x[:, 0]
        if self.numeric_projection is not None:
            numeric_rep = self.numeric_projection(x_nums)
            combined = torch.cat([cls_rep, numeric_rep], dim=1)
        else:
            combined = cls_rep
        out = self.mlp(combined).squeeze(-1)
        return out


def create_cat_mappings(df: pd.DataFrame, cat_cols: Sequence[str]) -> Dict[str, Dict[str, int]]:
    mappings: Dict[str, Dict[str, int]] = {}
    for col in cat_cols:
        unique_values = pd.Series(df[col].unique()).dropna().tolist()
        mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
        mappings[col]["__unknown__"] = len(unique_values)
    return mappings


def fit_tab_transformer(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    target_col: str,
    splitter: SpatialTemporalSplitter,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rmses: List[float] = []

    for train_idx, val_idx in splitter.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        scaler = RobustScaler()
        scaler.fit(train_df[num_cols])
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        train_scaled[num_cols] = scaler.transform(train_df[num_cols])
        val_scaled[num_cols] = scaler.transform(val_df[num_cols])

        cat_mappings = create_cat_mappings(train_df, cat_cols)

        train_dataset = TabularDataset(train_scaled, cat_cols, num_cols, train_scaled[target_col], cat_mappings)
        val_dataset = TabularDataset(val_scaled, cat_cols, num_cols, val_scaled[target_col], cat_mappings)

        cat_cardinalities = [len(cat_mappings[col]) - 1 for col in cat_cols]
        model = TabTransformer(cat_cardinalities, len(num_cols)).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = math.inf
        patience = 5
        patience_counter = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(epochs):
            model.train()
            for cats, nums, targets in train_loader:
                cats = cats.to(device)
                nums = nums.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                preds = model(cats, nums)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for cats, nums, targets in val_loader:
                    cats = cats.to(device)
                    nums = nums.to(device)
                    targets = targets.to(device)
                    preds = model(cats, nums)
                    val_loss = criterion(preds, targets)
                    val_losses.append(val_loss.item())
            avg_val_loss = float(np.mean(val_losses)) if val_losses else math.inf

            if avg_val_loss + 1e-4 < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        preds_list: List[float] = []
        targets_list: List[float] = []
        model.eval()
        with torch.no_grad():
            for cats, nums, targets in val_loader:
                cats = cats.to(device)
                nums = nums.to(device)
                preds = model(cats, nums)
                preds_list.extend(preds.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        preds_arr = np.array(preds_list)
        targets_arr = np.array(targets_list)
        rmse = math.sqrt(mean_squared_error(np.expm1(targets_arr), np.expm1(preds_arr)))
        rmses.append(rmse)

    return float(np.mean(rmses))


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Godina proizvodnje": "Godina_proizvodnje",
        "Zapremina motora": "Zapremina_motora",
        "Konjske snage": "Konjske_snage",
    }
    # also strip whitespace from column names
    df = df.rename(columns=lambda c: c.strip())
    return df.rename(columns=rename_map)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure numeric conversions
    df["Godina_proizvodnje"] = pd.to_numeric(df["Godina_proizvodnje"], errors="coerce").fillna(0).astype(int)
    current_year = datetime.now().year
    df["Starost"] = current_year - df["Godina_proizvodnje"]
    df["Starost"] = df["Starost"].clip(lower=0)
    # Assign region
    def assign_region(city: str) -> str:
        for region, cities in REGION_DEFINITIONS.items():
            if isinstance(city, str) and city in cities:
                return region
        return "Zapadna Srbija"

    df["Region"] = df["Grad"].apply(assign_region)

    # Estimate a 'nova cena' per brand as the 95th percentile observed price, fallback to median
    nova_cena = df.groupby("Marka")["Cena"].transform(lambda s: s.quantile(0.95) if len(s.dropna()) > 0 else np.nan)
    nova_cena = nova_cena.fillna(df["Cena"].median())

    # brand-specific alpha
    alpha = df["Marka"].map(ALPHA_BY_BRAND).fillna(1.1)
    df["Deprecijacioni_indeks"] = (nova_cena - df["Cena"]) / np.power(np.where(df["Starost"] == 0, 1, df["Starost"]), alpha)
    df["Deprecijacioni_indeks"] = df["Deprecijacioni_indeks"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def remove_outliers_isolation_forest(df: pd.DataFrame, features: Sequence[str], contamination: float = 0.02) -> pd.DataFrame:
    df_work = df.copy()
    # keep only rows without NaNs in the selected features for fitting
    fit_df = df_work[features].dropna()
    if fit_df.shape[0] < 10:
        return df_work
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(fit_df)
    preds = iso.predict(fit_df)
    retained_index = fit_df.index[preds == 1]
    return df_work.loc[retained_index].reset_index(drop=True)


def loess_soft_clip(df: pd.DataFrame, feature: str, reference: str, frac: float = 0.3, clip_sigma: float = 2.5) -> None:
    # ensure numeric
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df[reference] = pd.to_numeric(df[reference], errors="coerce")
    valid = df[[feature, reference]].dropna()
    if valid.empty:
        return
    try:
        smoothed = lowess(valid[feature], valid[reference], frac=frac, return_sorted=False)
    except Exception:
        return
    residuals = valid[feature] - smoothed
    scale = np.median(np.abs(residuals - np.median(residuals)))
    if scale == 0 or np.isnan(scale):
        return
    lower = smoothed - clip_sigma * scale
    upper = smoothed + clip_sigma * scale
    clipped = np.clip(valid[feature], lower, upper)
    # assign as float to avoid pandas dtype warnings
    df.loc[valid.index, feature] = clipped.astype(float)


def apply_loess_clipping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    loess_soft_clip(df, "Kilometraza", "Starost")
    loess_soft_clip(df, "Konjske_snage", "Starost")
    loess_soft_clip(df, "Cena", "Starost")
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def prepare_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = normalise_column_names(df)
    df = add_derived_features(df)
    # coerce numeric-like columns to float to avoid pandas dtype warnings when assigning clipped values
    for col in ["Cena", "Kilometraza", "Konjske_snage", "Zapremina_motora", "Godina_proizvodnje"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df = remove_outliers_isolation_forest(
        df,
        ["Cena", "Kilometraza", "Konjske_snage", "Zapremina_motora", "Godina_proizvodnje"],
    )
    df = apply_loess_clipping(df)
    df = fill_missing(df)
    for col in ["Kilometraza", "Konjske_snage", "Zapremina_motora"]:
        df[col] = np.log1p(df[col].clip(lower=0))
    df["Log_Cena"] = np.log1p(df["Cena"].clip(lower=0))
    return df


def evaluate_gradient_boosting(df: pd.DataFrame, regions: Sequence[str]) -> float:
    categorical_features = ["Marka", "Grad", "Karoserija", "Gorivo", "Menjac", "Region"]
    numerical_features = [
        "Godina_proizvodnje",
        "Zapremina_motora",
        "Kilometraza",
        "Konjske_snage",
        "Starost",
        "Deprecijacioni_indeks",
    ]

    X = df[categorical_features + numerical_features]
    y = df["Log_Cena"].values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ]
    )

    # default (full) param grid
    param_grid = {
        "regressor__n_estimators": [500, 800, 1000],
        "regressor__learning_rate": [0.01, 0.03, 0.05],
        "regressor__max_depth": [5, 6, 7],
        "regressor__subsample": [0.8, 0.9],
    }

    # if a 'quick' marker is attached to df.attrs, use a smaller grid for fast iteration
    if df.attrs.get("quick_run"):
        param_grid = {
            "regressor__n_estimators": [200, 400],
            "regressor__learning_rate": [0.03],
            "regressor__max_depth": [5],
            "regressor__subsample": [0.9],
        }

    splitter = SpatialTemporalSplitter(
        regions=regions,
        region_column="Region",
        year_column="Godina_proizvodnje",
        recent_year_threshold=2018,
    )

    n_jobs = -1
    if df.attrs.get("quick_run"):
        n_jobs = 1

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=splitter,
        n_jobs=n_jobs,
        scoring="neg_mean_squared_error",
        verbose=0,
    )
    grid_search.fit(X, y)

    best_estimator: Pipeline = grid_search.best_estimator_

    rmses: List[float] = []
    for train_idx, val_idx in splitter.split(X):
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]
        best_estimator.fit(X_train, y_train)
        preds = best_estimator.predict(X_val)
        rmse = math.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(preds)))
        rmses.append(rmse)

    return float(np.mean(rmses))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python main.py <path_to_dataset>")

    dataset_path = sys.argv[1]
    quick = "--quick" in sys.argv
    df = prepare_dataframe(dataset_path)
    if quick:
        df.attrs["quick_run"] = True

    region_order = tuple(df["Region"].dropna().unique())

    gbr_rmse = evaluate_gradient_boosting(df, region_order)

    splitter = SpatialTemporalSplitter(
        regions=region_order,
        region_column="Region",
        year_column="Godina_proizvodnje",
        recent_year_threshold=2018,
    )
    cat_cols = ["Marka", "Grad", "Karoserija", "Gorivo", "Menjac", "Region"]
    num_cols = [
        "Godina_proizvodnje",
        "Zapremina_motora",
        "Kilometraza",
        "Konjske_snage",
        "Starost",
        "Deprecijacioni_indeks",
    ]

    tab_transformer_rmse = fit_tab_transformer(df[cat_cols + num_cols + ["Log_Cena"]], cat_cols, num_cols, "Log_Cena", splitter, epochs=4, batch_size=32)

    print(f"GradientBoostingRegressor RMSE: {gbr_rmse:.2f}")
    print(f"TabTransformer RMSE: {tab_transformer_rmse:.2f}")


if __name__ == "__main__":
    main()
