#!/usr/bin/env python
# -*- coding: utf-8 -*-

def opus_test():
    """ Figure 1 in the Nodalida paper. """
    df = result_files_to_df(args.result_files, result_type='confidence')
    print(df)
    # exit()
    sns.set_theme(style="white")
    relplot = sns.relplot(
        kind="line",
        data=df,
        x='Compound divergence',
        y="BLEU",
        palette='blues',
    )
    relplot.set(xlabel='Training set compound divergence')
    relplot.set(ylim=(43, 45.7))
    relplot.fig.set_size_inches(4, 3)
    relplot.fig.savefig(args.output, dpi=400, bbox_inches = 'tight')

def all_vocabs():
    """ Figure 2 in the Nodalida paper. """
    df = result_files_to_df(args.result_files, result_type='confidence')
    sns.set_theme(style="white")
    relplot = sns.relplot(
        kind="line",
        data=df,
        x='Compound divergence',
        y="BLEU",
        hue='Vocab size',
        palette='rocket',
    )
    relplot.set(xlabel='Compound divergence')
    sns.move_legend(relplot, "upper left", bbox_to_anchor=(0.5, 1.0), ncol=1)
    relplot.fig.set_size_inches(4, 5)
    relplot.fig.savefig(args.output, dpi=400)

def subplot_seeds():
    """ Figure 3 in the Nodalida paper. """
    per_seed_per_vocab = result_files_to_df(args.result_files, result_type='confidence')
    seeds = [[11,22],[33,44],[55,66],[77,88]] # 4 rows of 2
    # seeds = [[11,22,33,44],[55,66,77,88]] # 2 rows of 4
    n_rows = len(seeds)
    n_cols = len(seeds[0])
    sns.set_theme(style="white")
    fig, axes = plt.subplots(n_rows, n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            seed = seeds[row][col]
            df = per_seed_per_vocab[per_seed_per_vocab["seed"] == str(seed)]
            relplot = sns.lineplot(
                data=df,
                x='Compound divergence',
                # y="BLEU",
                y="chrF2++",
                hue='Vocab size',
                palette='deep',
                ax=axes[row][col],
            )
            relplot.set(xlabel='Compound divergence')
    # fig.set_size_inches(23, 11) # 2 rows of 4
    fig.set_size_inches(11, 17) # 4 rows of 2
    fig.savefig(args.output, dpi=400, bbox_inches = 'tight')


def genbench():
    """ Figure in the GenBench2023 paper """
    import pandas as pd
    map_langs = {
        'de': 'German',
        'fr': 'French',
        'el': 'Greek',
        'fi': 'Finnish',
    }
    df = result_files_to_df(args.result_files, result_type='chrf2')
    print(df.to_string())

    seed_map = {'1': '11', '2': '22', '3': '33'}
    new_df = pd.DataFrame(columns=['tgt_lang', 'seed', 'Compound divergence', 'BLEU'])
    for lang in ['de', 'fr', 'el', 'fi']:
        for seed in ['1', '2', '3']:
            random_split = df[df["tgt_lang"] == str(lang)]
            random_split = random_split[random_split["seed"] == str(seed)]
            random_split = random_split[random_split["Compound divergence"] == 'Random split']
            # print(random_split.to_string())
            # get best BLEU of all Training steps
            # best_bleu = random_split['BLEU'].max()
            
            best_bleu = random_split['chrF2++'].max()
            new_df = new_df.append({'tgt_lang': lang, 'seed': seed_map[seed],
                                    'Compound divergence': 0.5,
                                    # 'BLEU': best_bleu},
                                    'chrF2++': best_bleu},
                                    ignore_index=True)
        for seed in ['11', '22', '33']:
            for cd in [0.0, 1.0]:
                sub_df = df[df["tgt_lang"] == str(lang)]
                sub_df = sub_df[sub_df["seed"] == str(seed)]
                sub_df = sub_df[sub_df["Compound divergence"] == cd]
                # print(sub_df.to_string())
                # get best BLEU of all Training steps
                # best_bleu = sub_df['BLEU'].max()
                best_bleu = sub_df['chrF2++'].max()
                print(f'Best BLEU for {lang}, {seed}, {cd}: {best_bleu}')
                new_df = new_df.append({'tgt_lang': lang, 'seed': seed,
                                        'Compound divergence': cd,
                                        # 'BLEU': best_bleu},
                                        'chrF2++': best_bleu},
                                        ignore_index=True)
    # avg bleu across seeds
    # new_df = new_df.groupby(['tgt_lang', 'Compound divergence']).mean()
    # print(new_df.to_string())
    
    # print(new_df.to_string())
    # new_df = new_df.drop(['BLEU'], axis=1)
    # print(new_df.to_string())
    
    
    # df_chrf = new_df.pivot(index=['tgt_lang', 'seed'],
    #                 columns='Compound divergence',
    #                 values=['chrF2++'])
    # print(df_chrf.to_string())
    # print(df_chrf.to_latex())
    
    
    langs = [['de', 'fr'], ['el', 'fi']]
    n_rows = len(langs)
    n_cols = len(langs[0])
    sns.set_theme(style="white")
    fig, axes = plt.subplots(n_rows, n_cols)
    
    for row in range(n_rows):
        for col in range(n_cols):
            lang = langs[row][col]
            # for seed in ['11', '22']:
            sub_df = new_df[new_df["tgt_lang"] == str(lang)]
            # sub_df = sub_df[sub_df["Compound divergence"] != 'Random split']
            relplot = sns.scatterplot(
                data=sub_df,
                # x='Training steps',
                # y="BLEU",
                y="chrF2++",
                x="Compound divergence",
                # style='seed',
                palette='deep',
                ax=axes[row,col],
            )
            relplot.set(title=f'Target language: {map_langs[lang]}')

            # set x range
            axes[row, col].set_xlim([-0.25, 1.25])
            # set x tics
            
            
            

    # remove y axis label
    # for col in [1,2,3]:
        # axes[col].set_ylabel('')

    axes[0,1].set_ylabel('')
    axes[1,1].set_ylabel('')
    axes[0,0].set_xlabel('')
    axes[0,1].set_xlabel('')
    axes[0,0].set_xticklabels([])
    axes[0,1].set_xticklabels([])
    axes[1,0].set_xticks([0.0, 0.5, 1.0])
    xlabels = ['0.0', 'Random split', '1.0']
    axes[1,0].set_xticklabels(xlabels)
    axes[1,1].set_xticks([0.0, 0.5, 1.0])
    axes[1,1].set_xticklabels(xlabels)

    # fig.set_size_inches(15, 3) # 1 row of 4
    fig.set_size_inches(8, 8) # 2 rows of 2
    fig.savefig(args.output, dpi=400, bbox_inches = 'tight')

if __name__ == '__main__':
    import argparse
    from df_utils import result_files_to_df
    import seaborn as sns
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_files',  nargs='*', type=str, required=True)
    parser.add_argument('--output', type=str, default='figures/untitled.png')
    parser.add_argument('--type', type=str, required=True)
    args = parser.parse_args()

    if args.type == 'opus_test':
        opus_test()
    elif args.type == 'all_vocabs':
        all_vocabs()
    elif args.type == 'subplot_seeds':
        subplot_seeds()
    elif args.type == 'genbench':
        genbench()
    else:
        raise ValueError('Unknown type: ' + args.type)
