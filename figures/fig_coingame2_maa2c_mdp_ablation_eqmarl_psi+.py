from pathlib import Path

FILEPATH = Path(__file__)
FIGURE_OUTDIR = FILEPATH.parent/FILEPATH.stem
FIGURE_OUTDIR.mkdir(parents=True, exist_ok=True) # Create.

#### Seaborn color palette.
# [0.2980392156862745, 0.4470588235294118, 0.6901960784313725]
# [0.8666666666666667, 0.5176470588235295, 0.3215686274509804]
# [0.3333333333333333, 0.6588235294117647, 0.40784313725490196]
# [0.7686274509803922, 0.3058823529411765, 0.3215686274509804]
# [0.5058823529411764, 0.4470588235294118, 0.7019607843137254]
# [0.5764705882352941, 0.47058823529411764, 0.3764705882352941]
# [0.8549019607843137, 0.5450980392156862, 0.7647058823529411]
# [0.5490196078431373, 0.5490196078431373, 0.5490196078431373]
# [0.8, 0.7254901960784313, 0.4549019607843137]
# [0.39215686274509803, 0.7098039215686275, 0.803921568627451]

series=[
    dict(
        key='$\\mathtt{eQMARL-}\Psi^{+}$-L2',
        blob='../experiment_output/coingame_maa2c_mdp_eqmarl_psi+_L2/20240803T190951/metrics-[0-9].json',
        color=[0.8666666666666667,0.5176470588235295,0.3215686274509804],
        zorder=1,
    ),
    dict(
        key='$\\mathtt{eQMARL-}\Psi^{+}$-L5',
        blob='../experiment_output/coingame_maa2c_mdp_eqmarl_psi+/20240501T152929/metrics-[0-9].json',
        color=[0.8549019607843137, 0.5450980392156862, 0.7647058823529411],
        zorder=2,
    ),
    dict(
        key='$\\mathtt{eQMARL-}\Psi^{+}$-L10',
        blob='../experiment_output/coingame_maa2c_mdp_eqmarl_psi+_L10/20240803T190951/metrics-[0-9].json',
        color=[0.3333333333333333,0.6588235294117647,0.40784313725490196],
        zorder=3,
    ),
]


metrics = [
    dict(
        key='undiscounted_reward',
        title='Score',
    ),
    dict(
        key='coins_collected',
        title='Coins Collected',
    ),
    dict(
        key='own_coins_collected',
        title='Own Coins Collected',
    ),
    dict(
        key='own_coin_rate',
        title='Own Coin Rate',
    ),
]



figures = [
    dict(
        type='subplots',
        kwargs=dict(
            figsize=[5.499999861629998, 3.399186852607058],
        ),
        ax=[
            dict(
                metric=m_dict['key'],
                xlabel='Epoch',
                ylabel=m_dict['title'],
                series=[
                    dict(
                        type='plot_with_errorbar',
                        key=d['key'],
                        label=d['key'],
                        color=d['color'],
                        plot_method='mean-rolling',
                        error_method='std',
                        plot_kwargs=dict(linewidth=1, zorder=len(series)+d.get('zorder', i)),
                        fill_kwargs=dict(alpha=0.4, linewidth=0.1, zorder=d.get('zorder', i)),
                    ) for i, d in enumerate(series)
                ] + ([
                    dict(
                        type='axhline',
                        y=25,
                        linestyle='--',
                        color='grey',
                        linewidth=1,
                    ),
                    dict(
                        type='axhline',
                        y=20,
                        linestyle='--',
                        color='grey',
                        linewidth=1,
                    ),
                ] if 'reward' in m_dict['key'] or 'coins_collected' in m_dict['key'] else []
                ) + ([
                    dict(
                        type='axhline',
                        y=1,
                        linestyle='--',
                        color='grey',
                        linewidth=1,
                    ),
                    dict(
                        type='axhline',
                        y=0.95,
                        linestyle='--',
                        color='grey',
                        linewidth=1,
                    ),
                ] if 'rate' in m_dict['key'] else []),
                yticks=(dict(
                    ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
                    labels=['0.0', '0.2', '0.4', '0.6', '0.8', '0.95', '1.0'],
                ) if 'rate' in m_dict['key'] else None),
                legend_kwargs=dict(
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.12),
                    ncol=4,
                    fancybox=True,
                    shadow=True,
                ),
                reports=([
                    dict(
                        type='y-threshold-max',
                        yvalue=0.8,
                    ),
                    dict(
                        type='y-threshold-max',
                        yvalue=0.95,
                    ),
                    dict(
                        type='y-threshold-max',
                        yvalue=1.0,
                    ),
                    dict(
                        type='y-max',
                    ),
                    dict(
                        type='y-mean',
                    ),
                    dict(
                        type='y-std',
                    ),
                    dict(
                        type='y-ci-0.95',
                    ),
                ] if 'rate' in m_dict['key'] else [
                    dict(
                        type='y-threshold-max',
                        yvalue=20,
                    ),
                    dict(
                        type='y-threshold-max',
                        yvalue=25,
                    ),
                    dict(
                        type='y-max',
                    ),
                    dict(
                        type='y-mean',
                    ),
                    dict(
                        type='y-std',
                    ),
                    dict(
                        type='y-ci-0.95',
                    ),
                ]),
            ),
        ],
        savefig_kwargs=dict(
            fname=FIGURE_OUTDIR/f"{FILEPATH.stem}-{m_dict['key']}.pdf",
            format='pdf',
            bbox_inches='tight',
        ),
        seaborn_style='ticks',
        seaborn_style_kwargs={},
        seaborn_context='paper',
        seaborn_context_kwargs={},
        # zoom_region=(dict(
        #     inset_axes=dict(
        #         bounds=[0.65, 0.15, 0.3, 0.3],
        #         xlim=[2500, 3000],
        #         ylim=[0.9, 1.01],
        #     ),
        #     indicate_inset_zoom=dict(
        #         edgecolor="black",
        #     ),
        # ) if 'rate' in m_dict['key'] # Zoom for rate metric.
        # else dict(
        #     inset_axes=dict(
        #         bounds=[0.65, 0.15, 0.3, 0.3],
        #         xlim=[2500, 3000],
        #         ylim=[20, 27],
        #     ),
        #     indicate_inset_zoom=dict(
        #         edgecolor="black",
        #     ),
        # )),
    ) for m_dict in metrics
]



__all__ = ['series', 'figures']