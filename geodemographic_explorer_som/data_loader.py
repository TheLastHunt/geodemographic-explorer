import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

def load_data(path: str) -> gpd.GeoDataFrame:
    """
    Read the GeoPackage file.
    """
    return gpd.read_file(path)


def scale_data(
    gdf: gpd.GeoDataFrame,
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Standard scale all numeric columns.
    Returns both the scaled and the original dataframes.

    """
    exclude_cols = exclude_cols or []

    # select numeric columns, minus any excludes
    numeric = gdf.select_dtypes(include=["number"]).copy()
    numeric.drop(columns=[c for c in exclude_cols if c in numeric.columns], inplace=True, errors='ignore')

    scaler = StandardScaler()
    scaled_arr = scaler.fit_transform(numeric.values)
    scaled_df = pd.DataFrame(scaled_arr, columns=numeric.columns, index=gdf.index)

    # carry through any grid coordinates
    for coord in ("hex_x", "hex_y"):
        if coord in gdf.columns:
            scaled_df[coord] = gdf[coord].values

    return scaled_df, gdf
