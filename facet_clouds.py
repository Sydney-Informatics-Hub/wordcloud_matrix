
def _freqs_from_csr(mat, row_idx, idx_to_vocab):
    start, stop = mat.indptr[row_idx:row_idx + 2]
    return list(zip((idx_to_vocab[x.astype(np.int64)]
                     for x in mat.indices[start:stop]),
                    mat.data[start:stop]))


def _circle_mask(height, width, cy, cx, r):
    dist = np.sqrt((np.arange(-cy, height - cy) ** 2).reshape(-1, 1) +
                   np.arange(-cx, width - cx) ** 2)
    return dist < r


def draw_cloud_circles(cloud, mat, centers, radii, labels=None, label_style='largest', margin=5,
                       background_kwargs=None, idx_to_vocab=None):
    """
    Parameters
    ----------
    cloud : wordcloud.WordCloud
    mat : scipy.sparse.csr_matrix
        Each row corresponds to a category plotted in a circle, except for an
        optional additional row, which is used to cloud the background (i.e. a
        background or reference distribution).
        Columns correspond to vocabulary items.  Values represent joint
        frequency or another measure of association.
    centers : array-like of (x, y) float pairs
        Where to plot the center of each circle.
    radii : array-like of floats
    labels : list of strings, optional
        Label for each circle
    margin : float
        Margin between circle clouds
    background_kwargs : dict, optional
        Settings to WordCloud for rendering the background
    idx_to_vocab : sequence/mapping int -> str, optional
        Mapping from column index in mat to term string.
        Default is spacy.load('en').vocab.strings.

    Returns
    -------
    cloud : wordcloud.WordCloud
        Not the same instance as was input
    indptr : list of int
        The number of entries in each cloud's contribution to layout_.
        The first entry corresponds to the input cloud, followed by
        an entry for each row in mat.
    """
    assert len(centers) == len(radii)
    assert mat.shape[0] in (len(centers), len(centers) + 1)
    assert mat.format == 'csr'
    if idx_to_vocab is None:
        idx_to_vocab = nlp.vocab.strings
    cloud = copy.copy(cloud)

    if not hasattr(cloud, 'layout_'):
        cloud.layout_ = []

    indptr = [len(cloud.layout_)]
    for i in range(mat.shape[0]):
        tmp_cloud = copy.copy(cloud)
        if i == len(centers):
            # background mask
            mask = np.zeros((cloud.height, cloud.width), dtype=bool)
            for j, ((cy, cx), r) in enumerate(zip(centers, radii)):
                mask += _circle_mask(cloud.height, cloud.width,
                                     cy, cx, r + margin)
            mask = ~mask
            if background_kwargs:
                for k, v in background_kwargs.items():
                    setattr(tmp_cloud, k, v)
        else:
            center = centers[i]
            radius = radii[i]
            mask = _circle_mask(cloud.height, cloud.width,
                                center[0], center[1],
                                radius)
            tmp_cloud.max_words = int(radius)

        freqs = _freqs_from_csr(mat, i, idx_to_vocab)
        tmp_cloud.mask = (~mask) * 255

        if len(freqs) and labels is not None and i < len(labels):
            if label_style == 'largest':
                # HACKY. DO NICER.
                tmp_cloud2 = copy.copy(tmp_cloud)
                tmp_cloud2.max_font_size = tmp_cloud2.height
                tmp_cloud2.mask = (~mask) * 255
                tmp_cloud2.color_func = lambda *args, **kwargs: "black"
                tmp_cloud2.prefer_horizontal = 1.0
                tmp_freqs = [('\n'.join(textwrap.wrap(labels[i], 15)), np.inf)]
                tmp_cloud2.generate_from_frequencies(tmp_freqs)
                cloud.layout_.extend(tmp_cloud2.layout_)

                tmp_cloud.mask = (~mask ^ (tmp_cloud2.to_array() < 255).any(axis=-1)) * 255

        if len(freqs):
            tmp_cloud.generate_from_frequencies(freqs)
            cloud.layout_.extend(tmp_cloud.layout_)
        indptr.append(len(cloud.layout_))

    return cloud, indptr


# TODO: init cloud and circles
def init_cloud_circles(n_circles, max_rad=150, border=30, per_row=3, margin=5, **cloud_kwargs):
    n_rows = int(np.ceil(float(n_circles) / per_row))
    cloud = wordcloud.WordCloud(max_font_size=max_rad // 5,
                                height=(n_rows * max_rad + border) * 2 + (n_rows - 1) * margin,
                                width=(per_row * max_rad + border) * 2 + (per_row - 1) * margin,
                                **cloud_kwargs)
    centers = list(zip(np.repeat(np.linspace(max_rad + border, cloud.height - max_rad - border, n_rows), per_row)[:n_circles],
                       np.tile(np.linspace(max_rad + border, cloud.width - max_rad - border, per_row), n_rows)[:n_circles].astype(int)))
    radii = np.log(counts) / np.log(counts.max()) * max_rad
    return cloud, centers, radii
