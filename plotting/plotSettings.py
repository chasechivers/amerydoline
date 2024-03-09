import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




# from rcmod.plotting_context
base_context = {
	"backend": 'TkAgg',

	"font.size": 12,
	"axes.labelsize": 12,
	"axes.titlesize": 12,
	"xtick.labelsize": 11,
	"ytick.labelsize": 11,
	"legend.fontsize": 11,

	"axes.axisbelow": True,
	"axes.linewidth": 1.25,
	"grid.linewidth": 1,
	"lines.linewidth": 1.5,
	"lines.markersize": 6,
	"patch.linewidth": 1,

	"xtick.major.width": 1.25,
	"ytick.major.width": 1.25,
	"xtick.minor.width": 1,
	"ytick.minor.width": 1,

	"xtick.major.size": 6,
	"ytick.major.size": 6,
	"xtick.minor.size": 4,
	"ytick.minor.size": 4,
	"xtick.direction": "in",
	"ytick.direction": "in",
	"xtick.top": True,
	"ytick.right": True,

	"font.family": ["sans-serif"],
	"font.sans-serif": ["Helvetica"],
	"text.latex.preamble": [r'\usepackage{siunitx}', r'\sisetup{detect-all}', r'\usepackage{helvet}',
	                        r'\usepackage{sansmath}', r'\sansmath'],
	'text.usetex': False,
	'mathtext.fontset': 'stixsans',
	"figure.dpi": 125.,
	"figure.autolayout": True,
	"figure.figsize": (6.35, 4.87),
	"legend.frameon": False
}

sns.set(style="ticks", palette="colorblind", color_codes=True,
        rc=base_context)

COLS = [np.append(np.asarray([col]), 1) for col in sns.color_palette()]

def _scale(dic, scaling):
	for k, v in dic.items():
		if isinstance(v, (bool, str, list, tuple)):
			dic[k] = v
		else:
			dic[k] = v * scaling
	return dic

def presentation(dark=True, scaling=1.3):
	scaled = {k: (v * scaling if isinstance(v, (float, int)) else v) for k,
	v in base_context.items()}
	scaled["figure.dpi"] = 100.
	if dark:
		plt.style.use('dark_background')
		scaled['figure.facecolor'] = 'black'
		scaled['figure.edgecolor'] = 'none'
	sns.set(style="ticks", font_scale=1.25, color_codes=True, palette="colorblind", rc=scaled)


def paper(scaling=1.2):
	scaled = _scale(base_context, scaling)
	scaled["figure.dpi"] = 125
	sns.set(style="ticks", font_scale=1.25, color_codes=True, palette="colorblind", rc=scaled)

def poster(scaling=1.3):
	scaled = {k: (v * scaling if not isinstance(v, (bool, str, list)) else v) for k, v in base_context.items()}
	scaled["figure.dpi"] = 150.
	scaled["lines.linewidth"] = 3
	scaled["axes.labelcolor"] = "black"
	scaled["axes.edgecolor"] = "black"
	scaled["xtick.color"] = "black"
	scaled["ytick.color"] = "black"
	scaled["savefig.transparent"] = True
	sns.set(style="ticks", font_scale=scaling, color_codes=True, palette="colorblind", rc=scaled)
