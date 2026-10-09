"""Decayed opponent tendencies, based only on completed public observations."""
DECAY = .85
MIN_ROUNDS = 3
MIN_PREFERENCE = .75
EARLY_TICKS = 45


def summarize_tendency(rounds):
    weights = {"A": 1., "B": 1.}  # A symmetric prior prevents one-round commitments.
    attacks, rushes = {"A": 0, "B": 0}, {"A": 0, "B": 0}
    inferred = {"A": 0, "B": 0}
    for age, row in enumerate(reversed(rounds)):
        public = {"L": "A", "R": "B"}.get(row.get("site"))
        rush = row.get("rush_site")
        pressure = row.get("pressure_site")
        side = public or rush or pressure
        if side not in weights:
            continue
        if public:
            attacks[public] += 1
        else:
            inferred[side] += 1
        if rush in rushes:
            rushes[rush] += 1
        reliability = 1. if public else .5
        weights[side] += DECAY ** age * reliability
        # Repeated early concentrations justify extra defensive coverage.
        if rush in weights:
            weights[rush] += DECAY ** age * .5
    probabilities = {s: weights[s] / sum(weights.values()) for s in weights}
    side = max(probabilities, key=probabilities.get)
    known = sum(attacks.values()) + sum(inferred.values())
    biased = known >= MIN_ROUNDS and probabilities[side] >= MIN_PREFERENCE
    allocation = {"A": 2, "Mid": 1, "B": 2}
    if biased:
        opposite = "B" if side == "A" else "A"
        allocation = {side: 3, opposite: 1, "Mid": 1}
    return dict(rounds=len(rounds), known_rounds=known, planted_attacks=attacks,
                observed_rushes=rushes, inferred_attacks=inferred,
                probabilities=probabilities, biased=biased,
                preferred_site=side if biased else None, allocation=allocation)


def allocation_for(tendency, current_contacts, model_site=None):
    """Fresh, concentrated sightings override a historical preference."""
    side = max(("A", "B"), key=lambda s: current_contacts.get(s, 0))
    opposite = "B" if side == "A" else "A"
    strong = current_contacts.get(side, 0) >= 2 and current_contacts.get(side, 0) > current_contacts.get(opposite, 0)
    if strong:
        return {side: 3, opposite: 1, "Mid": 1}, "current_contacts", side
    if tendency["biased"]:
        return dict(tendency["allocation"]), "round_history", tendency["preferred_site"]
    # A learned prediction alone cannot empty a site before any sighting.
    if model_site in ("A", "B") and current_contacts.get(model_site, 0) >= 1:
        other = "B" if model_site == "A" else "A"
        return {model_site: 3, other: 1, "Mid": 1}, "learned_with_contact", model_site
    return dict(A=2, Mid=1, B=2), "balanced", None


def scale_allocation(allocation, alive):
    """Distribute remaining players proportionally, retaining both sites."""
    if alive >= 5:
        return dict(allocation)
    if alive <= 1:
        side = max(("A", "B"), key=lambda s: allocation[s])
        return {"A": int(side == "A") * alive, "B": int(side == "B") * alive, "Mid": 0}
    result = {"A": 1, "B": 1, "Mid": 0}
    for _ in range(alive-2):
        side = max(("A", "B", "Mid"), key=lambda s: (allocation[s] * alive / 5 - result[s], allocation[s]))
        result[side] += 1
    return result
