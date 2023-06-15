from prince import MCA
import plotly.express as px


def plot_mca(df_toplot, fpath):
    mca = MCA(
        n_components=2,
        n_iter=30,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )

    fitted_mca = mca.fit(df_toplot.dropna())

    col_coords = fitted_mca.column_coordinates(df_toplot.dropna())
    col_coords.columns = [f"component {i}" for i in col_coords.columns]
    col_coords = col_coords.assign(
        variable=col_coords.index.name or "column",
        value=col_coords.index.astype(str),
    )

    col_coords['kind'] = col_coords.value.map(lambda x: "_".join(x.split('_')[:2]))
    # col_coords.value = col_coords.value.map(
    #     lambda x: "_".join(i for i in x.split('_')[3:] if i not in ['immigrati', 'italiani', 'emozioni']))
    fig = px.scatter(col_coords, x='component 0', y='component 1', text='value', color='kind',
                     labels={
                         'component 0': f"component 0 (frac. var: {mca.cumulative_percentage_of_variance_[0] / 100:.3})",
                         'component 1': f"component 1 (frac. var.: {mca.cumulative_percentage_of_variance_[1] / 100:.3})"}
                     )
    fig.for_each_trace(lambda t: t.update(textfont_color=t.marker.color, textposition='top center', textfont_size=12, ))
    # plotly.io.write_image(fig, fpath, format=format)
    fig.write_html(fpath)
    fig.show(figsize=(20, 20))
    return mca
    # fig.update_xaxes(
    #     range=[-0.25,.11],  # sets the range of xaxis
    # )
    # fig.update_yaxes(
    #     range=[-0.4,.35],  # sets the range of xaxis
    # )
