import numpy as np
import pandas as pd
import geopandas as gpd
import math

from minisom import MiniSom
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, GeoJSONDataSource,
    HoverTool, LinearColorMapper, ColorBar,
    FixedTicker, CustomJS, TableColumn, DataTable
)
from bokeh.models.glyphs import HexTile
from bokeh.palettes import Viridis256, Category10, Inferno256
from bokeh.events import ButtonClick
from typing import Tuple, List
from sklearn.cluster import AgglomerativeClustering


def build_hex_plot(
    hex_df: pd.DataFrame,
    som: MiniSom,
    um_flat: np.ndarray,
    node_cluster_labels: np.ndarray
):
    """
    Builds the Hex Plot and its respective Color Bars.

    """
    # Set up the data
    size = 1.0
    X, Y, _ = som.get_weights().shape
    umin, umax = um_flat.min(), um_flat.max()
    
    # Create the palettes
    n_clusters = int(hex_df['cluster'].max()) + 1
    base_palette = Category10[10]
    cluster_palette = (base_palette * ((n_clusters // 10) + 1))[:n_clusters]
    cmap_cl = LinearColorMapper(palette=cluster_palette, low=0, high=n_clusters-1)
    cmap_u = LinearColorMapper(palette=Viridis256, low=umin, high=umax)


    records = []
    for i in range(X):
        for j in range(Y):
            idx  = i * Y + j
            d    = um_flat[idx]
            norm = int((d - umin) / (umax - umin) * 255)
            records.append({
                'bmu_x': i,
                'bmu_y': j,
                'cluster': int(node_cluster_labels[idx]),
                'u_color': Viridis256[norm],
                'u_dist': d,
                'color': cluster_palette[int(node_cluster_labels[idx])],
                'alpha': 1.0,
            })

    node_df = pd.DataFrame.from_records(records)
    node_df['display_color'] = node_df['color']
    node_df["disp_q"] = node_df["bmu_x"]
    node_df["disp_r"] = (
        node_df["bmu_y"] - 0.5 * node_df["bmu_x"] + 0.5 * (node_df["bmu_x"] % 2)
    )
    source_hex = ColumnDataSource(node_df)

    # Create the figure
    p_hex = figure(
        title="SOM Grid",
        tools="pan,wheel_zoom,reset,tap",
        match_aspect=True,
        width=600, height=600,
        background_fill_color="#ffffff",
        outline_line_color="#cccccc",
        toolbar_location="above"
    )
    p_hex.title.text_align     = "left"
    p_hex.title.text_font_size = "16pt"
    p_hex.title.align          = "left"
    p_hex.grid.visible         = False
    p_hex.axis.visible         = False

    # Render the tiles
    hex_renderer = p_hex.hex_tile(
        q="disp_q", r="disp_r", size=size, orientation="flattop",
        source=source_hex,
        fill_color={"field": "display_color"},
        fill_alpha="alpha",
        line_color="#ffffff",
        line_width=0.5,
    )

    # Set up hover tooltips
    hover = HoverTool(
        renderers=[hex_renderer],
        tooltips=[
            ("Unit", "(@bmu_x, @bmu_y)"),
            ("Cluster", "@cluster"),
            ("U‐Dist", "@u_dist{0.0000}"),
        ],
        point_policy="follow_mouse"
    )
    p_hex.add_tools(hover)

    # Create Color Bars
    cb_cl = ColorBar(
        color_mapper=cmap_cl,
        ticker=FixedTicker(ticks=list(range(n_clusters))),
        major_label_overrides={i: str(i) for i in range(n_clusters)},
        label_standoff=12,
        border_line_color=None,
        location=(0,0)
    )
    
    cb_u   = ColorBar(
        color_mapper=cmap_u,
        label_standoff=12,
        location=(0,0),
        title="U-Matrix",
        visible=False
    )

    p_hex.add_layout(cb_cl, 'right')
    p_hex.add_layout(cb_u,  'right')

    return p_hex, source_hex, hex_renderer, cb_cl, cb_u

def build_map_plot(
    geo_df: gpd.GeoDataFrame,
    hex_df: pd.DataFrame,
    cluster_buttons: List
) -> Tuple[figure, GeoJSONDataSource]:
    """
    Creates the geographic map plot.

    """

    # Set up the data
    df = geo_df.copy()
    df['cluster'] = hex_df['cluster'].values

    source_map = GeoJSONDataSource(geojson=df.to_json())

    # Create color palettes
    max_c = int(df['cluster'].max())
    base = Category10[10]
    palette = (base * ((max_c // 10) + 1))[:max_c+1]
    cmap    = LinearColorMapper(palette=palette, low=0, high=max_c)

    # Create the figure
    p_map = figure(
        title="Lisbon Metropolitan Area Map",
        tools="pan,wheel_zoom,reset,tap",
        width=600, height=600, background_fill_color="#ffffff",
        toolbar_location="above"
    )

    p_map.title.text_align  = "left"    # horizontal centering
    p_map.title.text_font_size = "16pt"   # (optional) tweak your font size
    p_map.title.align = "left"

    p_map.axis.visible = False
    p_map.grid.visible = False

    # Render the patches
    patches = p_map.patches(
        'xs', 'ys', source=source_map,
        fill_color={'field': 'cluster', 'transform': cmap},
        line_color="white", line_width=0.5, hover_line_color="black"
    )
    p_map.add_tools(HoverTool(renderers=[patches], tooltips=[("BGRI2021","@BGRI2021"),
                                                             ("Cluster", "@cluster"),
                                                             ("Unit", "(@bmu_x, @bmu_y)")]))

    return p_map, source_map


def build_data_table(df: pd.DataFrame) -> DataTable:
    """
    Build a Data Table with one row per sample.
    """
    df = df.drop(columns=['geometry'], errors='ignore')
    source = ColumnDataSource(df)

    columns = []
    for colname in df.columns:
        columns.append(TableColumn(field=colname, title=colname, width=100))

    return DataTable(
        source=source,
        columns=columns,
        width=800,
        height=300,
        fit_columns=False,
        index_position=None
    )



def build_component_plane(
    hex_unit_df: pd.DataFrame,
    variable: str
) -> Tuple[figure, ColumnDataSource, LinearColorMapper, HoverTool]:
    """
    Build the Component Planes hex grid.
    """
    
    # Set up the data
    hex_unit_df["disp_q"] = hex_unit_df["bmu_x"]
    hex_unit_df["disp_r"] = (
        hex_unit_df["bmu_y"] - 0.5 * hex_unit_df["bmu_x"] + 0.5 * (hex_unit_df["bmu_x"] % 2)
    )
    source_cp = ColumnDataSource(hex_unit_df)

    # Create color palette
    low, high = float(hex_unit_df[variable].min()), float(hex_unit_df[variable].max())
    mapper_cp = LinearColorMapper(palette=Inferno256, low=low, high=high)

    # Create the figure
    p_cp = figure(
        title="Component Plane",
        tools="pan,wheel_zoom,reset,tap",
        match_aspect=True, width=600, height=600,
        background_fill_color="#ffffff", outline_line_color="#cccccc",
        toolbar_location="above"
    )
    p_cp.grid.visible = False
    p_cp.axis.visible = False

    p_cp.title.text_align  = "left"    # horizontal centering
    p_cp.title.text_font_size = "16pt"   # (optional) tweak your font size
    p_cp.title.align = "left"

    # Render the tiles
    renderer = p_cp.hex_tile(
        q="disp_q", r="disp_r", size=1, orientation="flattop",
        source=source_cp,
        fill_color={'field': variable, 'transform': mapper_cp},
        line_color="white", line_width=0.5
    )

    # Create the hover tooltips
    hover_cp = HoverTool(
        renderers=[renderer],
        tooltips=[("Value", f"@{variable}")]
    )
    p_cp.add_tools(hover_cp)

    # Generate the Color Bar
    cb_cp = ColorBar(color_mapper=mapper_cp, label_standoff=8, location=(0,0))
    p_cp.add_layout(cb_cp, 'right')

    return p_cp, source_cp, mapper_cp, hover_cp


def build_variable_choropleth(
    geo_with_bmu: gpd.GeoDataFrame,
    variable: str
) -> Tuple[figure, GeoJSONDataSource, LinearColorMapper, HoverTool]:
    df = geo_with_bmu.copy()
    source_ch = GeoJSONDataSource(geojson=df.to_json())
    low, high = float(df[variable].min()), float(df[variable].max())
    mapper_ch = LinearColorMapper(palette=Inferno256, low=low, high=high)

    p_ch = figure(
        title="Choropleth Map",
        tools="pan,wheel_zoom,reset,tap",
        width=600, height=600,
        background_fill_color="#ffffff",
        toolbar_location="above"
    )
    p_ch.axis.visible = False
    p_ch.grid.visible = False

    p_ch.title.text_align  = "left"    # horizontal centering
    p_ch.title.text_font_size = "16pt"   # (optional) tweak your font size
    p_ch.title.align = "left"

    patches = p_ch.patches(
        'xs', 'ys', source=source_ch,
        fill_color={'field': variable, 'transform': mapper_ch},
        line_color="white", line_width=0.5
    )

    hover_ch = HoverTool(
        renderers=[patches],
        tooltips=[("Value", f"@{variable}"),
                  ("BGRI2021","@BGRI2021"),
                  ("Unit", "(@bmu_x, @bmu_y)")]
    )
    p_ch.add_tools(hover_ch)

    cb_ch = ColorBar(color_mapper=mapper_ch, label_standoff=8, location=(0,0))
    p_ch.add_layout(cb_ch, 'right')

    return p_ch, source_ch, mapper_ch, hover_ch

def build_true_component_plane(
    som: MiniSom,
    variables: List[str],
    variable: str
):
    """
    Build a SOM component plane from the trained weights
    """
    # Flatten the SOM’s weight tensor
    X, Y, F = som.get_weights().shape
    flat_weights = som.get_weights().reshape(X*Y, F)
    node_df = pd.DataFrame(flat_weights, columns=variables)

    # Adjust grid coordinates
    node_df['bmu_x'] = np.repeat(np.arange(X), Y)
    node_df['bmu_y'] = np.tile(np.arange(Y), X)
    node_df['disp_q'] = node_df['bmu_x']
    node_df['disp_r'] = (
        node_df['bmu_y']
        - 0.5 * node_df['bmu_x']
        + 0.5 * (node_df['bmu_x'] % 2)
    )

    source_cp = ColumnDataSource(node_df)

    # Build color palette
    low, high = float(node_df[variable].min()), float(node_df[variable].max())
    mapper_cp = LinearColorMapper(palette="Inferno256", low=low, high=high)

    # Draw the figure
    p_cp = figure(
        title="Component Plane",
        tools="pan,wheel_zoom,reset,tap",
        match_aspect=True,
        width=600, height=600,
        background_fill_color="#ffffff",
        toolbar_location="above"
    )
    p_cp.grid.visible = False
    p_cp.axis.visible = False
    p_cp.title.text_align     = "left"
    p_cp.title.text_font_size = "16pt"
    p_cp.title.align          = "left"

    hex_renderer = p_cp.hex_tile(
        q="disp_q", r="disp_r", size=1, orientation="flattop",
        source=source_cp,
        fill_color={'field': variable, 'transform': mapper_cp},
        line_color="white"
    )

    # Hover tool
    hover_cp = HoverTool(
        renderers=[hex_renderer],
        tooltips=[("Value", f"@{variable}"),
                  ("Unit", "(@bmu_x, @bmu_y)")]
    )
    p_cp.add_tools(hover_cp)

    # Colorbar
    cb_cp = ColorBar(color_mapper=mapper_cp, label_standoff=8, location=(0,0))
    p_cp.add_layout(cb_cp, 'right')

    return p_cp, source_cp, mapper_cp, hover_cp