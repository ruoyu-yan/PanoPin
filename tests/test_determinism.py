def test_pin_sets_single_thread_and_seeds():
    import torch, numpy as np
    from panopin.determinism import pin
    pin(0)
    assert torch.get_num_threads() == 1
    a = np.random.rand(3)
    pin(0)
    b = np.random.rand(3)
    assert (a == b).all()   # same seed -> same numpy draw
