# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 21:30:35 2018

Adopted from: https://jwalton3141.github.io/Embed-Publication-Matplotlib-Latex/
Accessed: 21.08.2018
@author: brummli
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

textwidthThesis=419.16621#pt   #for current standard setting
#resulting Height=259.058963982pt

#textwidthBeamer = 335.99*#pt
#resulting Height=207.6532398800772pt

#Rescaled because of title making height to large
textwidthBeamer = 285.5#pt
#resulting Height=176.44pt


nice_fonts = {
        # Use LaTex to write all text
        "text.usetex": True,
        #CHANGED FOR BEAMER TO SANS SERIF AND ALL SIZES+1
        "font.family": "sans-serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 11,
        "font.size": 11,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
}

def set_size(width, fraction=1,subplot=[1, 1]):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplot[0] / subplot[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

#How to use examples
#fig, ax = plt.subplots(1, 1, figsize=set_size(textwidthThesis))

if __name__ == '__main__':
    mpl.rcParams.update(nice_fonts)
    
    x = np.linspace(0, 2*np.pi, 100)
    # Initialise figure instance
    fig, ax = plt.subplots(1, 1, figsize=set_size(textwidthBeamer))
    # Plot
    ax.plot(x, np.sin(x))
    ax.set_xlim(0, 2*np.pi)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\sin{(\theta)}$')
    
    # Save and remove excess whitespace
    plt.savefig('plots/thesisReady/testFig_1.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig)