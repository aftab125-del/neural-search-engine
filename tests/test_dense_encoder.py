import numpy as np
import pytest
from neuralsearch.core.dense_encoder import DenseEncoder


@pytest.fixture(scope="module")
def encoder():
    return DenseEncoder()


def test_dense_encoder_dimensions_and_norm(encoder):
    text = "Neural information retrieval with dense vectors"
    vec = encoder.encode(text)

    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    # Verify L2 unit norm
    norm = np.linalg.norm(vec)
    assert pytest.approx(norm, 0.001) == 1.0


def test_dense_encoder_semantic_similarity(encoder):
    v_car = encoder.encode("a modern luxury car driving on a highway")
    v_auto = encoder.encode("an automobile traveling along the road")
    v_recipe = encoder.encode("baking chocolate chip cookies with fresh butter and flour")

    sim_synonyms = float(np.dot(v_car, v_auto))
    sim_unrelated = float(np.dot(v_car, v_recipe))

    # Synonyms should have high positive similarity, unrelated should be near zero
    assert sim_synonyms > 0.50
    assert sim_unrelated < 0.20
    assert sim_synonyms - sim_unrelated > 0.40


def test_dense_encoder_batch_consistency(encoder):
    texts = [
        "First document about artificial intelligence.",
        "Second document regarding machine learning algorithms.",
    ]
    batch_vecs = encoder.encode_batch(texts)
    v0 = encoder.encode(texts[0])
    v1 = encoder.encode(texts[1])

    assert batch_vecs.shape == (2, 384)
    # Under INT8 dynamic quantization, batch and single encoding maintain >98% cosine alignment
    assert float(np.dot(batch_vecs[0], v0)) > 0.98
    assert float(np.dot(batch_vecs[1], v1)) > 0.98
