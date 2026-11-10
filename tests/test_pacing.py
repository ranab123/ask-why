"""Pacing behaviour of the selectors on synthetic sequences (no browser, no LLM)."""
from askwhy.config import Lambdas, TauParams
from askwhy.recorder import StepRecord
from askwhy.selector.base import ObserverOut
from askwhy.selector.random_sel import RandomSelector
from askwhy.selector.scored import ScoredSelector


def _step(t, action_type="click", name="Some link", sig="page", label="navigate", boundary=False):
    return StepRecord(t=t, url_before="", page_sig=sig, action=f"{action_type}('a{t}')", action_type=action_type,
                      target_bid=f"a{t}", target_name=name, activity_label=label, boundary=boundary)


def _obs(match_bid, prob, action_type="click", confidence=0.9, alts=()):
    return ObserverOut(predicted=[{"action_type": action_type, "bid": match_bid, "target": "", "prob": prob}], ok=True,
                       interp_ok=True, confidence=confidence, alt_rationales=list(alts))


def run(sel, steps, observers):
    fires = []
    for s, o in zip(steps, observers):
        d = sel.score(s, o)
        if d.fire:
            fires.append(s.t)
        sel.observe_outcome(s)
    return fires


def test_all_high_spends_exactly_budget_and_respects_gap():
    sel = ScoredSelector(Lambdas(), TauParams())
    sel.start(est_decisions=12, budget=2)
    steps = [_step(t, name="Delete", sig=f"p{t}") for t in range(12)]  # every step critical + novel → a decision point
    obs = [_obs("zzz", 0.9, confidence=0.2, alts=["a", "b"]) for _ in range(12)]  # surprising, uncertain, ambiguous
    fires = run(sel, steps, obs)
    assert len(fires) == 2
    assert fires[1] - fires[0] > TauParams().min_gap


def test_boring_clicks_are_not_decision_points():
    sel = ScoredSelector(Lambdas(), TauParams())
    sel.start(est_decisions=8, budget=2)
    steps = [_step(t, action_type="scroll", name="", sig="same") for t in range(20)]
    obs = [_obs(f"a{t}", 0.95, "scroll") for t in range(20)]
    decisions = [sel.score(s, o).is_decision for s, o in zip(steps, obs)]
    assert not any(decisions)
    assert sel.budget_left == 2  # nothing spent on scrolls


def test_late_spike_gets_asked():
    sel = ScoredSelector(Lambdas(), TauParams())
    sel.start(est_decisions=8, budget=1)
    steps = [_step(t, action_type="scroll", name="", sig="same") for t in range(18)] + [_step(18, name="Delete", sig="list"), _step(19, name="Update", sig="form")]
    obs = [_obs(f"a{t}", 0.95, "scroll") for t in range(18)] + [_obs("zzz", 0.5, confidence=0.4, alts=["x"]), _obs("zzz", 0.5, confidence=0.4)]
    fires = run(sel, steps, obs)
    assert fires == [18]


def test_threshold_moves_with_pace():
    sel = ScoredSelector(Lambdas(), TauParams())
    sel.start(est_decisions=12, budget=4)
    tau_start = sel.threshold()
    sel.budget_left, sel.decisions_seen = 1, 2  # spent 3 of 4 after 2 decisions → ahead of pace → higher τ
    assert sel.threshold() > tau_start
    sel.budget_left, sel.decisions_seen = 4, 9  # spent nothing after 9 decisions → behind → lower τ
    assert sel.threshold() < tau_start


def test_random_selector_spends_budget_on_decision_points_only():
    sel = RandomSelector(TauParams(), seed=1)
    sel.start(est_decisions=10, budget=3)
    steps = []
    for t in range(30):
        steps.append(_step(t, name="Delete", sig="list") if t % 3 == 0 else _step(t, action_type="scroll", name="", sig="list"))
    fires = run(sel, steps, [None] * 30)
    assert len(fires) == 3
    assert all(t % 3 == 0 for t in fires)
