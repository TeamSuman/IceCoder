import numpy as np
from sklearn.cluster import KMeans
def Ticker(ax):
    """
    Styles the matplotlib axis..
    
    """
    from matplotlib.ticker import AutoMinorLocator
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    minor_locator = AutoMinorLocator(2)
    ax.yaxis.set_minor_locator(minor_locator)
    return ax

def plotFreeNRG(data, title = None, colors = "black"):
    #fig, ax = plt.subplots(ncols=1, nrows=2,gridspec_kw={'height_ratios':[15,1]},figsize = [10, 10])
    #z,x,y = np.histogram2d(test_cv[:,0],test_cv[:,1], bins=75)
    z,x,y = np.histogram2d(data[:,0], data[:,1], bins=150)

    # compute free energies
    F = -np.log(z)
    F = F - np.min(F)
    # contour plot
    extent = [x[0], x[-1], y[0], y[-1]]
    return z, extent

def PlotCenters(data, num_center, color):
    kmeans = KMeans(n_clusters=num_center, random_state=0).fit(data)
    centers = kmeans.cluster_centers_
    for i in range(num_center):
        plt.scatter(x = centers[i][0], y = centers[i][1],marker = "*", s = 200, lw = 0.5, c = color )

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out