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
        key='eQMARL',
        blob='~/Downloads/output/coingame_maa2c_quantum_pomdp_nnreduce_4qubits/20240419T112012/metrics-[0-5].json',
        color=[0.2980392156862745,0.4470588235294118,0.6901960784313725],
    ),
    dict(
        key='Classical Central CTDE',
        blob='~/Downloads/output/coingame_maa2c_classical_pomdp_central/20240501T185443/metrics-[0-5].json',
        color=[0.8666666666666667,0.5176470588235295,0.3215686274509804],),
    dict(
        key='Classical Split CTDE',
        blob='~/Downloads/output/coingame_maa2c_classical_pomdp/20240418T133536/metrics-[0-5].json',
        color=[0.3333333333333333,0.6588235294117647,0.40784313725490196],
    ),
    
    
    
    #### ANCILLARY
    
    # dict(
    #     key='Q 15 layers',
    #     blob='~/Downloads/output/coingame4_maa2c_quantum_pomdp_nnreduce_4qubits_15layers/20240501T120947/metrics-[0-5].json',
    #     color=[0.5058823529411764, 0.4470588235294118, 0.7019607843137254],
    # ),
    
    # # coingame_maa2c_quantum_pomdp/20240418T140242/
    # dict(
    #     key='eQMARL (pure quantum)',
    #     blob='~/Downloads/output/coingame_maa2c_quantum_pomdp/20240418T140242//metrics-[0-5].json',
    #     color=[0.39215686274509803, 0.7098039215686275, 0.803921568627451],
    # ),
    
    # dict(
    #     key='eQMARL ($\Psi^{+}$)',
    #     blob='~/Downloads/output/coingame_maa2c_quantum_pomdp_nnreduce_4qubits_psi+/20240502T131044/metrics-[0-5].json',
    #     color=[0.8, 0.7254901960784313, 0.4549019607843137],
    # ),
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
                        key=d['key'],
                        label=d['key'],
                        color=d['color'],
                        plot_method='mean-rolling',
                        error_method='minmax-rolling',
                        plot_kwargs=dict(linewidth=1),
                        fill_kwargs=dict(alpha=0.4, linewidth=0.2)
                    ) for d in series
                ],
                legend_kwargs=dict(loc='lower right'),
            ),
        ],
        savefig_kwargs=dict(
            fname=FIGURE_OUTDIR/f"{FILEPATH.stem}-{m_dict['key']}.pdf",
            format='pdf',
            bbox_inches='tight',
        ),
    ) for m_dict in metrics
]



__all__ = ['series', 'figures']