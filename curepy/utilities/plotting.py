# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import logging
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

__all__ = ["plot_corner", "hist2d", "quantile"]


def plot_corner(
    xs,
    bins: int = 20,
    range=None,
    weights=None,
    color: str = "k",
    smooth=None,
    smooth1d=None,
    ticks=None,
    ticklabels=None,
    labels=None,
    label_kwargs: dict = None,
    show_titles: bool = False,
    title_fmt: str = ".2f",
    title_kwargs: dict = None,
    truths=None,
    truth_color: str = "#4682b4",
    scale_hist: bool = False,
    quantiles=None,
    verbose: bool = False,
    fig=None,
    max_n_ticks: int = 5,
    top_ticks: bool = False,
    use_math_text: bool = False,
    hist_kwargs: dict = None,
    **hist2d_kwargs,
):
    """
    Make a corner plot showing the projections of a data set in a
    multi-dimensional space.  Remaining keyword arguments are forwarded to
    :func:`hist2d` or used for ``matplotlib`` styling.

    :param xs: The samples.  Must be a 1-D or 2-D array where the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.
    :param bins: Number of bins to use in histograms, either as a fixed
        value for all dimensions or as a list of integers for each
        dimension.
    :param range: A list where each element is either a length-2 tuple
        containing lower and upper bounds, or a float in ``(0, 1)`` giving
        the fraction of samples to include in the bounds.
    :param weights: The weight of each sample.  If ``None`` (default),
        samples are given equal weight.
    :param color: A ``matplotlib`` colour for all histograms.
    :param smooth: Standard deviation for Gaussian kernel smoothing of the
        2-D histograms.  ``None`` disables smoothing.
    :param smooth1d: Standard deviation for Gaussian kernel smoothing of
        the 1-D histograms.  ``None`` disables smoothing.
    :param ticks: Custom tick positions for each dimension.
    :param ticklabels: Custom tick labels for each dimension.
    :param labels: A list of names for the dimensions.  Defaults to
        ``DataFrame`` column names when ``xs`` is a ``pandas.DataFrame``.
    :param label_kwargs: Extra keyword arguments passed to
        ``set_xlabel`` / ``set_ylabel``.
    :param show_titles: If ``True``, display the 0.5 quantile with
        1-sigma errors as a title above each 1-D histogram.
    :param title_fmt: Format string for quantile values in titles.
    :param title_kwargs: Extra keyword arguments passed to ``set_title``.
    :param truths: Reference values to indicate on the plots.  Individual
        values may be ``None``.
    :param truth_color: ``matplotlib`` colour for the truth markers.
    :param scale_hist: If ``True``, scale 1-D histograms so the zero line
        is visible.
    :param quantiles: Fractional quantiles to show as vertical dashed
        lines on 1-D histograms.
    :param verbose: If ``True``, print the computed quantile values.
    :param fig: Existing ``matplotlib.Figure`` to overplot onto.
    :param max_n_ticks: Maximum number of axis ticks per axis.
    :param top_ticks: If ``True``, label ticks at the top of each axis.
    :param use_math_text: If ``True``, render very large or small axis
        tick exponents as powers of 10.
    :param hist_kwargs: Extra keyword arguments forwarded to the 1-D
        histogram plots.
    :returns: The ``matplotlib.Figure`` containing the corner plot.
    """
    if quantiles is None:
        quantiles = []
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    # Try filling in labels from pandas.DataFrame columns.
    if labels is None:
        try:
            labels = xs.columns
        except AttributeError:
            pass

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], (
        "I don't believe that you want more " "dimensions than samples!"
    )

    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")

    # Parse the parameter ranges.
    if range is None:
        if "extents" in hist2d_kwargs:
            logging.warn(
                "Deprecated keyword argument 'extents'. " "Use 'range' instead."
            )
            range = hist2d_kwargs.pop("extents")
        else:
            range = [[x.min(), x.max()] for x in xs]
            # Check for parameters that never change.
            m = np.array([e[0] == e[1] for e in range], dtype=bool)
            if np.any(m):
                raise ValueError(
                    (
                        "It looks like the parameter(s) in "
                        "column(s) {0} have no dynamic range. "
                        "Please provide a `range` argument."
                    ).format(", ".join(map("{0}".format, np.arange(len(m))[m])))
                )

    else:
        # If any of the extents are percentiles, convert them to ranges.
        # Also make sure it's a normal list.
        range = list(range)
        for i, _ in enumerate(range):
            try:
                emin, emax = range[i]
            except TypeError:
                q = [0.5 - 0.5 * range[i], 0.5 + 0.5 * range[i]]
                range[i] = quantile(xs[i], q, weights=weights)

    if len(range) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in range]
    except TypeError:
        if len(bins) != len(range):
            raise ValueError("Dimension mismatch between bins and range")

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0  # size of one side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # w/hspace size
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    if fig is None:
        fig, axes = pl.subplots(K, K, figsize=(dim * 1.2, dim))
        # Format the figure.

    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(
        left=lb, bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace
    )

    # Set up the default histogram keywords.
    if hist_kwargs is None:
        hist_kwargs = dict()
    hist_kwargs["color"] = hist_kwargs.get("color", color)
    if smooth1d is None:
        hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")

    for i, x in enumerate(xs):

        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            ax = axes[i, i]
        # Plot the histograms.
        if smooth1d is None:
            n, _, _ = ax.hist(
                x, bins=bins[i], weights=weights, range=np.sort(range[i]), **hist_kwargs
            )
        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")
            n, b = np.histogram(
                x, bins=bins[i], weights=weights, range=np.sort(range[i])
            )
            n = gaussian_filter(n, smooth1d)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
            ax.plot(x0, y0, **hist_kwargs)

        if truths is not None and truths[i] is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=color)

            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

        if show_titles:
            title = None
            if title_fmt is not None:
                # Compute the quantiles for the title. This might redo
                # unneeded computation but who cares.
                q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84], weights=weights)
                q_m, q_p = q_50 - q_16, q_84 - q_50

                # Format the quantile display.
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))

                # Add in the column name if it's given.
                if labels is not None:
                    title = "{0} = {1}".format(labels[i], title)

            elif labels is not None:
                title = "{0}".format(labels[i])

            if title is not None:
                ax.set_title(title, **title_kwargs)

        # Set up the axes.
        ax.set_xlim(range[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i], fontsize=18, **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)

            # use MathText for axes ticks
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=use_math_text))

        for j, y in enumerate(xs):

            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                ax = axes[i, j]

            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                if ticks:
                    ax.set_xticks(ticks[j])
                    if j == K - 1:
                        ax.set_xticklabels(ticklabels[j], fontsize=10)
                        [l.set_rotation(90) for l in ax.get_xticklabels()]
                continue

            # Deal with masked arrays.
            if hasattr(y, "compressed"):
                y = y.compressed()

            hist2d(
                y,
                x,
                fig,
                ax=ax,
                range=[range[j], range[i]],
                weights=weights,
                color=color,
                smooth=smooth,
                bins=[bins[j], bins[i]],
                **hist2d_kwargs,
            )

            if truths is not None:
                if truths[i] is not None and truths[j] is not None:
                    ax.plot(truths[j], truths[i], "s", color=truth_color)
                if truths[j] is not None:
                    ax.axvline(truths[j], color=truth_color)
                if truths[i] is not None:
                    ax.axhline(truths[i], color=truth_color)

            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                if ticks:
                    ax.set_xticks(ticks[j])
                    ax.set_xticklabels(ticklabels[j], fontsize=10)
                    [l.set_rotation(90) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[j], fontsize=18, **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)

            if j > 0:
                ax.set_yticklabels([])
            else:
                if ticks:
                    ax.set_yticks(ticks[i])
                    ax.set_yticklabels(ticklabels[i], fontsize=10)
                ax.set_ylabel(labels[i], fontsize=18, **label_kwargs)
                ax.yaxis.set_label_coords(-0.3, 0.5)
    pl.tight_layout()

    return fig


def quantile(
    x,
    q,
    weights=None,
):
    """
    Compute sample quantiles with support for weighted samples.

    .. note::
        When ``weights`` is ``None``, this method simply calls
        :func:`numpy.percentile` with the values of ``q`` multiplied by 100.

    :param x: The samples.
    :param q: The list of quantiles to compute.  All values must be in
        the range ``[0, 1]``.
    :param weights: An optional weight corresponding to each sample.
    :returns: The sample quantiles computed at ``q``.
    :raises ValueError: If any value in ``q`` is outside ``[0, 1]``, or
        if the lengths of ``x`` and ``weights`` do not match.
    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, 100.0 * q)
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()


def hist2d(
    x,
    y,
    fig,
    bins: int = 20,
    range=None,
    weights=None,
    levels=None,
    smooth=None,
    ax=None,
    color: str = None,
    plot_datapoints: bool = True,
    plot_density: bool = True,
    plot_contours: bool = True,
    no_fill_contours: bool = False,
    fill_contours: bool = False,
    contour_kwargs: dict = None,
    contourf_kwargs: dict = None,
    data_kwargs: dict = None,
    **kwargs,
):
    """
    Plot a 2-D histogram of samples.

    :param x: Samples for the horizontal axis.
    :param y: Samples for the vertical axis.
    :param fig: ``matplotlib.Figure`` to which the colour-bar axes are added.
    :param bins: Number of bins for the 2-D histogram.
    :param range: Axis ranges ``[[x_min, x_max], [y_min, y_max]]``.
    :param weights: Per-sample weights.
    :param levels: Contour levels to draw.
    :param smooth: Standard deviation for Gaussian kernel smoothing.
    :param ax: ``matplotlib.Axes`` instance on which to draw.  Defaults
        to the current active axes.
    :param color: ``matplotlib`` colour for the plot elements.
    :param plot_datapoints: If ``True``, draw the individual data points.
    :param plot_density: If ``True``, render the density colour map.
    :param plot_contours: If ``True``, draw the contour lines.
    :param no_fill_contours: If ``True``, suppress the white fill beneath
        contours.
    :param fill_contours: If ``True``, fill the contours.
    :param contour_kwargs: Extra keyword arguments forwarded to
        ``axes.contour``.
    :param contourf_kwargs: Extra keyword arguments forwarded to
        ``axes.contourf``.
    :param data_kwargs: Extra keyword arguments forwarded to ``axes.plot``
        when drawing the individual data points.
    """
    if ax is None:
        ax = pl.gca()

    # Set the default range based on the data range if not provided.
    if range is None:
        if "extent" in kwargs:
            logging.warn(
                "Deprecated keyword argument 'extent'. " "Use 'range' instead."
            )
            range = kwargs["extent"]
        else:
            range = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This is the color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)]
    )

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2
    )

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels) + 1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins, weights=weights)
    except ValueError:
        raise ValueError(
            "It looks like at least one of your sample columns "
            "have no dynamic range. You could try using the "
            "'range' argument."
        )

    if smooth is not None:
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
        ]
    )
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
        ]
    )

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.05)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    """if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)
    """
    """if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)"""

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.

    if plot_density:
        density_cmap = "gnuplot_r"
        im = ax.pcolor(
            X, Y, H.T / np.sum(H), cmap=density_cmap, alpha=0.5, vmin=0, vmax=0.01
        )

    cax = fig.add_axes([0.8, 0.65, 0.05, 0.3])
    # img = pl.imshow(np.random.rand(20,20)*0.2, cmap='gnuplot_r')
    # img.set_visible(False)
    colb = pl.colorbar(im, cax=cax)
    colb.set_label("P", fontsize=20, rotation=0, labelpad=20)
    # Plot the contour edge colors.
    """if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)"""

    ax.set_xlim(range[0])
    ax.set_ylim(range[1])
