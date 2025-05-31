from pandas import Series
from pandas import DataFrame
import numpy as np
import plotly.graph_objects as go

def plot_confussion(
        confussion_table: DataFrame,
        vertical_column: str,
        horizontal_column: str,
        label_column: str,
        value_column: str,
        vertical_ascending: bool = False,
        horizontal_ascending: bool = False,
        title: str = "Marimekko Confusion Matrix"
    ) -> None:
    """
    This function creates a Marimekko plot from a confusion matrix.
    """
    marimekko_table = confussion_table.groupby([vertical_column, horizontal_column])[[value_column,label_column]].agg({
        value_column: lambda x: sum(x),
        label_column: lambda y: ', '.join(y)
    }).reset_index()
    marimekko_groups = marimekko_table.groupby([horizontal_column])[[value_column]].agg('sum').reset_index().rename(columns={value_column: 'width'})
    marimekko_table = marimekko_table.merge(marimekko_groups, on=horizontal_column)
    marimekko_table['height'] = marimekko_table[value_column] / marimekko_table['width']
    marimekko_table = marimekko_table.sort_values(by=horizontal_column, ascending=horizontal_ascending)
    grouped = marimekko_table.groupby(vertical_column).agg({
        'width': list,
        'height': list,
        value_column: list,
        label_column: list,
        horizontal_column: list
    })
    grouped = grouped.sort_index(ascending=vertical_ascending)
    fig = go.Figure()
    row = grouped.iloc[0]
    widths = Series(row['width'])
    h_tags = Series(row[horizontal_column])
    for row in grouped.iloc:
        group = row.name
        heights = row['height']
        values = row[value_column]
        labels = row[label_column]
        fig.add_trace(go.Bar(name = group,
                        y = heights,
                        x = np.cumsum(widths) - widths,
                        customdata=np.transpose([labels, values]),
                        width = widths,
                        offset = 0,
                        texttemplate = "<br>".join([
                            group,
                            "%{customdata[0]}",
                            "%{customdata[1]}",
                            "%{y:.0%}"
                        ]),
                        hovertemplate = "<br>".join([
                            group,
                            "%{customdata[0]}",
                            "%{customdata[1]}",
                            "%{y:.4%}"
                        ])
                    ))
    fig.layout.yaxis.tickformat = ',.0%'   
    fig.update_layout(barmode = "relative")
    fig.update_yaxes(range=[0,1])
    fig.update_xaxes(range=[0,np.sum(widths)])
    fig.update_xaxes(
        tickvals=np.cumsum(widths)-widths/2,
        ticktext= ["%s<br>%d" % (l, w) for l, w in zip(h_tags, widths)]
    )
    fig.update_layout(title=title)
    fig.show()
