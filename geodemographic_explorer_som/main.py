import os
import random
import pandas as pd
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import CustomJS, Spacer, Tabs, TabPanel, Select, Button, Div, ColumnDataSource, AutocompleteInput, GeoJSONDataSource
from bokeh.events import ButtonClick, Tap

from data_loader import load_data, scale_data
from som_model import train_som, compute_umatrix
from cluster_analysis import assign_clusters, compute_cluster_means
from widgets import create_cluster_buttons, create_variable_selector, create_hex_mode_selector
from plots import build_hex_plot, build_map_plot, build_data_table, build_component_plane, build_variable_choropleth, build_true_component_plane

random.seed(0)
np.random.seed(0)

# Data Loading
HERE = os.path.dirname(__file__)
gpkg_path = os.path.join(HERE, 'data', 'mydata.gpkg')
gdf = load_data(gpkg_path)

exclude_cols = [
    'OBJECTID', 'BGRI2021', 'SSNUM21', 'SECSSNUM21', 'SUBSECCAO',
    'DT21', 'DTMN21', 'DTMNFR21', 'DTMNFRSEC21', 'SECNUM21',
    'NUTS1', 'NUTS2', 'NUTS3', 'SHAPE_Length', 'SHAPE_Area'
]

scaled_df, geo_df = scale_data(gdf, exclude_cols=exclude_cols)

# SOM training and clustering
som     = train_som(scaled_df)
um_flat = compute_umatrix(som)

hex_df, node_labels = assign_clusters(som, scaled_df)
cluster_means_df    = compute_cluster_means(hex_df)

# Create plots and widgets
hex_mode_selector = create_hex_mode_selector()
cluster_buttons = create_cluster_buttons(n_clusters=cluster_means_df.shape[0])

p_hex, source_hex, hex_renderer, cb_cl, cb_u = build_hex_plot(
    hex_df, som, um_flat, node_cluster_labels=node_labels
)

def update_hex_mode(attr, old, new):
    # new is either "Cluster" or "U-Matrix"
    useU = (new == "U-Matrix")
    # copy-then-assign to force a full data sync
    d = dict(source_hex.data)
    d["display_color"] = d["u_color"] if useU else d["color"]
    source_hex.data = d
    # show/hide the colorbars
    cb_cl.visible = not useU
    cb_u.visible = useU

hex_mode_selector.on_change("value", update_hex_mode)

geo_with_bmu = geo_df.assign(bmu_x=hex_df['bmu_x'], bmu_y=hex_df['bmu_y'], cluster=hex_df['cluster'])
p_map, source_map = build_map_plot(geo_with_bmu, hex_df, cluster_buttons)

raw_numeric = [
    c for c in geo_df.select_dtypes(include=['number']).columns
    if c not in ('hex_x','hex_y')
]

input_vars = list(scaled_df.drop(columns=["hex_x", "hex_y"], errors="ignore").columns)

hex_unit_means = (
    geo_with_bmu
    .groupby(['bmu_x','bmu_y'])[input_vars]
    .mean()
    .reset_index()
)

X, Y, F = som.get_weights().shape
flat_weights = som.get_weights().reshape(X*Y, F)

node_df = pd.DataFrame(flat_weights, columns=input_vars)

node_df['bmu_x'] = np.repeat(np.arange(X), Y)
node_df['bmu_y'] = np.tile(np.arange(Y), X)
node_df['disp_q'] = node_df['bmu_x']
node_df['disp_r'] = (
    node_df['bmu_y']
    - 0.5 * node_df['bmu_x']
    + 0.5 * (node_df['bmu_x'] % 2)
)

source_comp = ColumnDataSource(node_df)

initial = input_vars[0]

p_comp, source_comp, mapper_comp, hover_comp = build_true_component_plane(som, input_vars, initial)
p_choro, source_choro, mapper_choro, hover_choro = build_variable_choropleth(geo_with_bmu, initial)

selector = create_variable_selector(input_vars)

mins = {v: float(hex_unit_means[v].min()) for v in input_vars}
maxs = {v: float(hex_unit_means[v].max()) for v in input_vars}

region_mins = { v: float(hex_unit_means[v].min()) for v in input_vars }
region_maxs = { v: float(hex_unit_means[v].max()) for v in input_vars }

weight_mins = { v: float(node_df[v].min()) for v in input_vars }
weight_maxs = { v: float(node_df[v].max()) for v in input_vars }

# Define JS callback for interactivity
callback = CustomJS(args=dict(
    selector=selector,
    comp_renderer=p_comp.renderers[0],
    choro_renderer=p_choro.renderers[0],
    mapper_comp=mapper_comp,
    mapper_choro=mapper_choro,
    hover_comp=hover_comp,
    hover_choro=hover_choro,
    region_mins=region_mins, region_maxs=region_maxs,
    weight_mins=weight_mins, weight_maxs=weight_maxs
), code="""
    const varname = selector.value;

    // swap the fill field on both renderers
    comp_renderer.glyph.fill_color.field  = varname;
    choro_renderer.glyph.fill_color.field = varname;

    // set each mapper’s domain from the correct table
    const lo_comp   = weight_mins[varname], hi_comp   = weight_maxs[varname];
    const lo_choro  = region_mins[varname], hi_choro  = region_maxs[varname];
    mapper_comp.low   = lo_comp;   mapper_comp.high   = hi_comp;
    mapper_choro.low  = lo_choro;  mapper_choro.high  = hi_choro;

    // update tooltips (unchanged)
    hover_comp.tooltips  = [["Value", "@" + varname]];
    hover_choro.tooltips = [["Value", "@" + varname]];

    // emit changes
    comp_renderer.glyph.change.emit();
    choro_renderer.glyph.change.emit();
    mapper_comp.change.emit();
    mapper_choro.change.emit();
    hover_comp.change.emit();
    hover_choro.change.emit();
""")

selector.js_on_change('value', callback)

# Build Data Table
data_table   = build_data_table(geo_with_bmu)
table_source = data_table.source

orig_table_data = { k: list(v) for k, v in table_source.data.items() }

bgr_codes = sorted(table_source.data['BGRI2021'])

search_input = AutocompleteInput(
    title="Search BGRI2021:",
    completions=bgr_codes,
    width=250
)

# Define more JS callbacks
reset_btn = Button(label="Reset", width=120)

reset_btn.js_on_event('button_click', CustomJS(args=dict(
    hex_src    = source_hex,
    map_src    = source_map,
    table_src  = table_source,
    original   = orig_table_data
), code="""
    // 1) clear all selections on hex & map
    hex_src.selected.indices = [];
    map_src.selected.indices = [];

    // 2) restore every column in the DataTable to its original full-array
    table_src.data = original;

    // 3) emit the change so the table redraws
    table_src.change.emit();
"""))

search_input.js_on_change('value', CustomJS(args=dict(
    hex_src   = source_hex,
    map_src   = source_map,
    table_src = table_source
), code="""
    const val = this.value;
    // 1) if empty → clear selections & reset
    if (!val) {
        hex_src.selected.indices = [];
        map_src.selected.indices = [];
        hex_src.selected.change.emit();
        map_src.selected.change.emit();
        return;
    }

    // 2) find the index in the TABLE CDS
    const codes = table_src.data['BGRI2021'];
    let row = codes.indexOf(val);
    if (row < 0) return;  // no match

    // 3) grab its BMU coords
    const bx = table_src.data['bmu_x'][row];
    const by = table_src.data['bmu_y'][row];

    // 4) find matching hex cells
    const hex_inds = [];
    const hx = hex_src.data['bmu_x'];
    const hy = hex_src.data['bmu_y'];
    for (let j = 0; j < hx.length; j++) {
        if (hx[j] === bx && hy[j] === by) {
            hex_inds.push(j);
        }
    }

    // 5) find matching map features (same row index in table_src → same feature index)
    const map_inds = [ row ];

    // 6) select & emit
    hex_src.selected.indices = hex_inds;
    map_src.selected.indices = map_inds;
    hex_src.selected.change.emit();
    map_src.selected.change.emit();
"""))

callback_table = CustomJS(args=dict(
    hex_src=source_hex,
    map_src=source_map,
    table_src=table_source
), code="""
    // build an array [0,1,2,…] for the “no selection” case
    function all_indices(n) {
        const a = [];
        for (let i = 0; i < n; i++) a.push(i);
        return a;
    }

    let inds = [];
    const hx_sel = hex_src.selected.indices;
    const mp_sel = map_src.selected.indices;

    if (hx_sel.length > 0) {
        // take the first (and only) SOM‐cell, get its bmu_x/y
        const i0 = hx_sel[0];
        const bx = hex_src.data['bmu_x'][i0];
        const by = hex_src.data['bmu_y'][i0];

        // find all map‐rows with that same BMU
        for (let j = 0; j < map_src.data['bmu_x'].length; j++) {
            if (map_src.data['bmu_x'][j] === bx
             && map_src.data['bmu_y'][j] === by) {
                inds.push(j);
            }
        }
    }
    else if (mp_sel.length > 0) {
        // direct map‐click: just mirror those rows
        inds = mp_sel.slice();
    }
    else {
        // nothing selected: show everything
        inds = all_indices(map_src.data['bmu_x'].length);
    }

    // rebuild table data from the chosen indices
    const new_data = {};
    for (const key in map_src.data) {
        new_data[key] = inds.map(i => map_src.data[key][i]);
    }
    table_src.data = new_data;
    table_src.change.emit();
""")
source_hex.selected.js_on_change('indices', callback_table)
source_map.selected.js_on_change('indices', callback_table)


for i, btn in enumerate(cluster_buttons):
    btn.js_on_event('button_click', CustomJS(args=dict(
        hex_src   = source_hex,
        map_src   = source_map,
        table_src = table_source,
        cl        = i
    ), code="""
        // 1) Gather the indices of all hex‐tiles in cluster cl
        const hex_inds = [];
        const hx_data  = hex_src.data['cluster'];
        for (let j = 0; j < hx_data.length; j++) {
            if (hx_data[j] === cl) hex_inds.push(j);
        }

        // 2) Gather the indices of all map‐regions in cluster cl
        const map_inds = [];
        const mx_data  = map_src.data['cluster'];
        for (let j = 0; j < mx_data.length; j++) {
            if (mx_data[j] === cl) map_inds.push(j);
        }

        // 3) Check if this cluster is already selected (toggle‐off)
        const already = hex_src.selected.indices.length === hex_inds.length
                     && hex_src.selected.indices.every((v, idx) => v === hex_inds[idx]);

        if (already) {
            // clear both selections
            hex_src.selected.indices = [];
            map_src.selected.indices = [];
        } else {
            // select the cluster on both plots
            hex_src.selected.indices = hex_inds;
            map_src.selected.indices = map_inds;
        }

        // 4) Emit the selection change events so callback_table runs
        hex_src.selected.change.emit();
        map_src.selected.change.emit();
    """))


source_hex.selected.js_on_change('indices', CustomJS(args=dict(
    hex_src=source_hex,
    map_src=source_map,
    table_src=table_source
), code="""
    const inds = hex_src.selected.indices;
    if (inds.length === 0) {
        map_src.selected.indices   = [];
        table_src.selected.indices = [];
    } else if (inds.length === 1) {
        const idx = inds[0];
        // find all regions with matching BMU coords
        const bx = hex_src.data['bmu_x'][idx];
        const by = hex_src.data['bmu_y'][idx];
        const xs = map_src.data['bmu_x'];
        const ys = map_src.data['bmu_y'];
        const region_inds = [];
        for (let i = 0; i < xs.length; i++) {
            if (xs[i] === bx && ys[i] === by) {
                region_inds.push(i);
            }
        }
        map_src.selected.indices   = region_inds;
        // highlight cluster row
        const cl = hex_src.data['cluster'][idx];
        table_src.selected.indices = [cl];
    } else {
        return;
    }
    map_src.change.emit();
    table_src.change.emit();
"""))


source_map.selected.js_on_change('indices', CustomJS(args=dict(
    hex_src=source_hex,
    map_src=source_map,
    table_src=table_source
), code="""
    const inds = map_src.selected.indices;
    if (inds.length === 0) {
        hex_src.selected.indices   = [];
        table_src.selected.indices = [];
    } else if (inds.length === 1) {
        const idx = inds[0];
        const bx  = map_src.data['bmu_x'][idx];
        const by  = map_src.data['bmu_y'][idx];
        const xs  = hex_src.data['bmu_x'];
        const ys  = hex_src.data['bmu_y'];
        let hex_i = null;
        for (let i = 0; i < xs.length; i++) {
            if (xs[i] === bx && ys[i] === by) { hex_i = i; break; }
        }
        hex_src.selected.indices   = hex_i !== null ? [hex_i] : [];
        const cl = map_src.data['cluster'][idx];
        table_src.selected.indices = [cl];
    } else {
        return;
    }
    hex_src.change.emit();
    table_src.change.emit();
"""))


p_hex.js_on_event(Tap, CustomJS(args=dict(src=source_hex), code="""
    const inds = src.selected.indices;
    if (inds.length === 1) {
        const idx = inds[0];
        if (src.data._last_sel === idx) {
            src.selected.indices = [];
            src.data._last_sel = null;
        } else {
            src.data._last_sel = idx;
        }
        src.selected.change.emit();
    }
"""))

p_map.js_on_event(Tap, CustomJS(args=dict(src=source_map), code="""
    const inds = src.selected.indices;
    if (inds.length === 1) {
        const idx = inds[0];
        if (src.data._last_sel === idx) {
            src.selected.indices = [];
            src.data._last_sel = null;
        } else {
            src.data._last_sel = idx;
        }
        src.selected.change.emit();
    }
"""))


source_comp.selected.js_on_change('indices', CustomJS(args=dict(
    comp_src=source_comp,
    map_src=source_choro
), code="""
    const inds = comp_src.selected.indices;
    if (inds.length === 0) {
        // no cell selected → clear map
        map_src.selected.indices = [];
    }
    else if (inds.length === 1) {
        // single cell: find all regions with same (bmu_x,bmu_y)
        const idx = inds[0];
        const bx = comp_src.data['bmu_x'][idx];
        const by = comp_src.data['bmu_y'][idx];
        const xs = map_src.data['bmu_x'];
        const ys = map_src.data['bmu_y'];
        const sel = [];
        for (let j = 0; j < xs.length; j++) {
            if (xs[j] === bx && ys[j] === by) {
                sel.push(j);
            }
        }
        map_src.selected.indices = sel;
    }
    // emit change so the glyph knows to update
    map_src.change.emit();
"""))


source_choro.selected.js_on_change('indices', CustomJS(args=dict(
    map_src=source_choro,
    comp_src=source_comp
), code="""
    const inds = map_src.selected.indices;
    if (inds.length === 0) {
        // no region selected → clear component‐plane
        comp_src.selected.indices = [];
    }
    else if (inds.length === 1) {
        // single region: find the matching plane cell
        const idx = inds[0];
        const bx = map_src.data['bmu_x'][idx];
        const by = map_src.data['bmu_y'][idx];
        const xs = comp_src.data['bmu_x'];
        const ys = comp_src.data['bmu_y'];
        let comp_idx = null;
        for (let j = 0; j < xs.length; j++) {
            if (xs[j] === bx && ys[j] === by) {
                comp_idx = j;
                break;
            }
        }
        comp_src.selected.indices = comp_idx !== null ? [comp_idx] : [];
    }
    comp_src.change.emit();
"""))

buttons_title = Div(text="<b>Select cluster:</b>", width=200)

# Attach custom titles to widgets
cluster_panel = column(
    row(Spacer(sizing_mode="stretch_width"), buttons_title, Spacer(sizing_mode="stretch_width")),
    row(*cluster_buttons, sizing_mode="scale_width"),
    sizing_mode="scale_width",
)

hex_selector_title = Div(text="<b>Select Hex Grid Mode</b>", width=200)


hex_selector_panel = column(
    row(Spacer(sizing_mode="stretch_width"), hex_selector_title, Spacer(sizing_mode="stretch_width")),
    row(hex_mode_selector, sizing_mode="scale_width"),
    sizing_mode="scale_width",
)

variable_selector_title = Div(text="<b>Select Varibale</b>", width=200)


variable_selector_panel = column(
    row(Spacer(sizing_mode="stretch_width"), variable_selector_title, Spacer(sizing_mode="stretch_width")),
    row(selector, sizing_mode="scale_width"),
    sizing_mode="scale_width",
)

# Assemble layouts
layout_1 = column(
    row(hex_selector_panel, Spacer(sizing_mode="stretch_width"), cluster_panel, Spacer(sizing_mode="stretch_width"), search_input, Spacer(sizing_mode="stretch_width"), reset_btn, margin = (10, 0, 10, 0), sizing_mode="scale_width"),
    row(p_hex, p_map, margin = (10, 0, 10, 0), sizing_mode="scale_width"),
    row(Spacer(sizing_mode="stretch_width"), data_table, Spacer(sizing_mode="stretch_width"), sizing_mode="scale_width", margin = (30, 0, 10, 0)),
    sizing_mode="scale_width",
    margin = (10, 5, 10, 5),
    name="layout_1"
)

layout_2 = column(
    row(Spacer(sizing_mode="stretch_width"), variable_selector_panel, Spacer(sizing_mode="stretch_width"), sizing_mode="scale_width", margin = (30, 0, 10, 0)),
    row(p_comp, p_choro, margin=(10,0,10,0), sizing_mode="scale_width"),
    sizing_mode="scale_width",
    margin=(10,5,10,5),
    name="layout_2"
)


# Add the layouts to the main application in tab form and add to root
tab1 = TabPanel(child=layout_1,   title="Clusters")
tab2 = TabPanel(child=layout_2, title="Single Variable")
tabs = Tabs(tabs=[tab1, tab2], sizing_mode="stretch_both", name="tabs")

curdoc().add_root(tabs)
curdoc().title = "SOM Dashboard"
