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


def get_fermi(fname):
    with open(fname) as myfile:
        for line in myfile:
            m = re.search('E-fermi\s*\:\s*(.*)\XC', line)
            if m is not None:
                return float(m.group(1))


def read_band(fname='wannier90.up_band.dat', fermi_shift=None):
    """
    plot the wannier band.

    :param fname: the filename of the _band.dat
    :param fermi_shift: (None | float) if None, no shift, else,  all the energies are shifted down by fermi_shift.
    """
    kslist = []
    ks = []
    ekslist = []
    eks = []
    wkslist = []
    wks = []
    have_weight = False
    with open(fname) as myfile:
        for line in myfile:
            if len(line) > 5:
                vals = map(float, line.strip().split())
                ks.append(vals[0])
                if fermi_shift is None:
                    eks.append(vals[1])
                else:
                    eks.append(vals[1] - fermi_shift)
                if len(vals) == 3:
                    have_weight = True
                    wks.append(vals[2])

            else:
                kslist.append(ks)
                ekslist.append(eks)
                wkslist.append(wks)
                ks = []
                eks = []
                wks = []
    if have_weight:
        return kslist, ekslist, wkslist
    else:
        return kslist, ekslist


def plot_band_from_data(es, kpts=None, efermi=None, labels=None):
    """
    plot the band.
    :param es: energies, nkpts*nbands array.
    """
    plt.cla()
    nbands, nkpts = np.asarray(es).shape
    if kpts is None:
        kpts = np.arrzy(range(nkpts))
    if len(np.asarray(kpts).shape) == 2:
        kpts = kpts[0]
    xmax = max(kpts)
    plt.xlim([0, xmax])
    if efermi is not None:
        plt.axhline(y=0.0, linestyle='--', color='blue')
        es = np.asarray(es) - efermi
    for e in es:
        plt.plot(kpts, e)

    if labels is not None:
        plt.xticks(labels[0], labels[1])
        for x in labels[0]:
            plt.axvline(x=x, color='gray')
    plt.show()


def plot_band(fname='wannier90.up_band.dat', efermi=None):
    """
    plot the band with data read from wannier90 output data.
    """
    plt.clf()
    kslist, ekslist = read_band(fname, fermi_shift=efermi)[:-1]
    xmax = max(kslist[0])
    plt.xlim([0, xmax])
    for ks, eks in zip(kslist, ekslist):
        plt.plot(ks, eks)
    plt.show()


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
                    linewidths=[4] * len(x),
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


def plot_band_weight_file(fname='wannier90.up_band.dat',
                          efermi=None,
                          weight=True,
                          yrange=None,
                          output=None,
                          style='alpha',
                          color='blue',
                          axis=None,
                          width=10):
    """
    plot the band with projection
    """
    #plt.cla()
    if weight:
        kslist, ekslist, wkslist = read_band(fname, fermi_shift=efermi)
    else:
        kslist, ekslist = read_band(fname, fermi_shift=efermi)[:2]
        wkslist = None
    if fname[:-3].endswith('band.'):
        xticks = read_xtics(fname[:-3] + 'gnu')
    else:
        xticks = read_xtics(fname[:fname.rfind('_')] + '.gnu')

    return plot_band_weight(
        kslist,
        ekslist,
        wkslist=wkslist,
        efermi=efermi,
        yrange=yrange,
        output=output,
        style=style,
        color=color,
        axis=axis,
        width=width,
        xticks=xticks)


def read_xtics(fname='wannier90.up_band.gnu'):
    text = open(fname).read()
    m = re.search('xtics\s*\((.*)\)\n', text)
    ts = m.group(1).split(',')
    names = []
    xs = []
    for t in ts:
        r = re.search(r'"\s*(.*)\s*"', t).group(1)
        x = float(t.strip().split()[-1])
        #print r,x
        names.append(r)
        xs.append(x)
    return names, xs


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
    #plot_band()
    #plot_band_weight(efermi=9.05)
    #print read_xtics()
