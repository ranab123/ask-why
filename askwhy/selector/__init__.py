from .base import Decision, ObserverOut, Selector
from .none_sel import NoneSelector
from .random_sel import RandomSelector


def make_selector(kind: str, cfg, est_decisions: int, budget: int) -> Selector:
    if kind == "none" or budget <= 0:
        sel: Selector = NoneSelector()
    elif kind == "random":
        sel = RandomSelector(cfg.tau, seed=cfg.seed or 0)
    elif kind == "scored":
        from .scored import ScoredSelector

        sel = ScoredSelector(cfg.lambdas, cfg.tau)
    else:
        raise ValueError(kind)
    sel.start(est_decisions, budget)
    return sel


__all__ = ["Decision", "ObserverOut", "Selector", "make_selector", "NoneSelector", "RandomSelector"]
