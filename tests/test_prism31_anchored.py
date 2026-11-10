import numpy as np
from prism.gene_calling.prism31.anchored import AnchoredMixture


def test_rare_code_centre_stays_in_its_own_slot():
    rng = np.random.default_rng(72)
    anchors = np.array([[0., 0.], [1., 0.], [0., 1.]])
    x = np.concatenate([rng.normal(centre, .12, (n, 2)) for centre, n in
                        zip([[.26, 0.], [.84, 0.], [0., 1.]], [12000, 60, 500])])
    model = AnchoredMixture(anchors, .15).fit(x)
    assert np.all(np.linalg.norm(model.means - anchors, axis=1) <= .150001)
    assert model.weights[0] > 50 * model.weights[1]
    assert np.array_equal(model.classify(anchors)[0], np.arange(3))
    assert np.isfinite(model.classify(x)[2]).all()
