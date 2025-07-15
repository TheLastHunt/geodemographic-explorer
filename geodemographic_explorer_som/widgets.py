from bokeh.models import Toggle, Button, Select, Div
from bokeh.palettes import Category10
from typing import List


def create_cluster_buttons(n_clusters: int) -> List[Button]:
    """
    Generate a list of Buttons for selecting clusters.

    The number of buttons is given by the number of clusters passed to the function.

    """
    base = Category10[10]
    palette = (base * ((n_clusters // 10) + 1))[:n_clusters]

    buttons: List[Button] = []
    for i in range(n_clusters):
        btn = Button(label=str(i), css_classes=[f"cluster-btn-{i}"], width=30)
        buttons.append(btn)
    return buttons

def create_variable_selector(vars: List[str]) -> Select:
    """
    Dropdown to pick which variable drives the plots in the "Single Variable" tab.

    """
    return Select(value=vars[0],
                  options=vars)

def create_hex_mode_selector() -> Select:
    """
    Dropdown to pick whether the hex‐plot is colored by Cluster or by U-Matrix.

    """
    return Select(
        value="Clusters",
        options=["Clusters", "U-Matrix"]
    )