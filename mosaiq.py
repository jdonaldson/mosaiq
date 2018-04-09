import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

#default colormaps for categorical/linear data
CAT_COLORMAP = plt.cm.get_cmap("Pastel2", 8)
LIN_COLORMAP = plt.cm.get_cmap("RdYlGn", 8)

def topn(dat, col, top=7):
    top_labels = dat[col].value_counts().sort_values(ascending=False)[0:top].index
    topmod = dat[col].map(lambda label: "NA_TOPN" if label not in top_labels else label)
    return topmod

def topbin(dat,col):
    if dat[col].dtype == "object":
        return topn(dat, col).reset_index(drop=True)
    _, bins = np.histogram(dat[col], bins=8)
    buckets = bins[np.digitize(dat[col], bins)-1].round(2).astype("str")
    col_label = col + " Bins"
    return pd.Series(data=buckets, name=col_label).reset_index(drop=True)


def mosaiq(dat, feature, target, invert_colors=False, cmap= None):
    _, axs = plt.subplots(figsize=(16, 9))
    datsrt = dat.sort_values([target, feature], ascending=[True, True])
    feat = topbin(datsrt, feature)
    targ = topbin(datsrt, target)
    datmos = pd.DataFrame({feature : feat, target : targ})

    def lab(var):
        return var[1]

    def props(var):
        value = var[1]
        if dat[target].dtype == "object":
            ordered = targ.value_counts().sort_values(ascending=False).index.tolist()

            colormap = CAT_COLORMAP if cmap is None else cmap
        else:
            ordered = targ.value_counts() \
                .index.astype("float") \
                .sort_values(ascending=False).tolist()

            ordered = [str(o) for o in ordered]
            colormap = LIN_COLORMAP if cmap is None else cmap

        if invert_colors:
            color_dict = {v : len(ordered) - i -1  for i, v in enumerate(ordered)}
        else:
            color_dict = {v : i  for i, v in enumerate(ordered)}
        def adj_trans(col):
            return (col[0], col[1], col[2], .5)

        return {"color" : adj_trans(colormap(color_dict[value]))}

    mosaic(datmos, [feature, target], properties = props, labelizer=lab, title="%s vs. %s" % (feature, target),ax=axs)

