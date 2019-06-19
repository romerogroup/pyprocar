#!/usr/bin/env python
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
import numpy as np
import re
import sys
import argparse
import os.path

def plot_band_weight(kslist,
                     ekslist,
                     wkslist=None,
                     efermi=None,
                     shift_efermi=False,
                     yrange=None,
                     output=None,
                     style='alpha',
                     color='blue',
                     axis=None,
                     width=10,
                     fatness=4,
                     xticks=None,
                     cmap=mpl.cm.bwr,
                     weight_min=-0.1,
                     weight_max=0.6):
    if axis is None:
        fig, a = plt.subplots()
    else:
        a = axis
    if efermi is not None and shift_efermi:
        ekslist = np.array(ekslist) - efermi
    else:
        ekslist = np.array(ekslist)

    xmax = max(kslist[0])
    if yrange is None:
        yrange = (np.array(ekslist).flatten().min() - 0.66,
                  np.array(ekslist).flatten().max() + 0.66)
    if wkslist is not None:
        for i in range(len(kslist)):
            x = kslist[i]
            y = ekslist[i]
            #lwidths=np.ones(len(x))
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            if style == 'width':
                lwidths = np.array(wkslist[i]) * width
                lc = LineCollection(segments, linewidths=lwidths, colors=color)
            elif style == 'alpha':
                lwidths = np.array(wkslist[i]) * width
                lc = LineCollection(
                    segments,
                    linewidths=[fatness] * len(x),
                    colors=[
                        colorConverter.to_rgba(
                            color, alpha=lwidth / (width + 0.001))
                        for lwidth in lwidths
                    ])
            elif style == 'color' or style == 'colormap':
                lwidths = np.array(wkslist[i]) * width
                norm = mpl.colors.Normalize(vmin=weight_min, vmax=weight_max)
                #norm = mpl.colors.SymLogNorm(linthresh=0.03,vmin=weight_min, vmax=weight_max)
                m = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                #lc = LineCollection(segments,linewidths=np.abs(norm(lwidths)-0.5)*1, colors=[m.to_rgba(lwidth) for lwidth in lwidths])
                lc = LineCollection(
                    segments,
                    linewidths=lwidths,
                    colors=[m.to_rgba(lwidth) for lwidth in lwidths])
            a.add_collection(lc)
    if axis is None:
        for ks, eks in zip(kslist, ekslist):
            a.plot(ks, eks, color='gray', linewidth=0.01)
        #a.set_xlim(0, xmax)
        #a.set_ylim(yrange)
        if xticks is not None:
            a.set_xticks(xticks[1])
            a.set_xticklabels(xticks[0])
            for x in xticks[1]:
                a.axvline(x, alpha=0.6, color='black', linewidth=0.7)
        if efermi is not None:
            if shift_efermi:
                a.axhline(linestyle='--', color='black')
            else:
                a.axhline(efermi, linestyle='--', color='black')

    return a


def main():
    parser = argparse.ArgumentParser(description='plot wannier bands.')
    parser.add_argument('fname', type=str, help='dat filename')
    parser.add_argument(
        '-e', '--efermi', type=float, help='Fermi energy', default=None)
    parser.add_argument(
        '-o', '--output', type=str, help='output filename', default=None)
    parser.add_argument(
        '-w',
        '--weight',
        action='store_true',
        help='use -w to plot weighted band.')
    parser.add_argument(
        '-y',
        '--yrange',
        type=float,
        nargs='+',
        help='range of yticks',
        default=None)
    parser.add_argument(
        '-s',
        '--style',
        type=str,
        help='style of line, width | alpha',
        default='width')
    args = parser.parse_args()
    if args.output is None:
        output = os.path.splitext(args.fname)[0] + '.png'
    if args.efermi is None:
        efermi = get_fermi('SCF/OUTCAR')
    plot_band_weight_file(
        fname=args.fname,
        efermi=efermi,
        weight=args.weight,
        yrange=args.yrange,
        style=args.style)
    if output is not None:
        plt.savefig(output)

    plt.show()


if __name__ == '__main__':
    main()
