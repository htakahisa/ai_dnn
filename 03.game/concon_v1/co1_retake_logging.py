"""Readable console summaries; structured checkpoint/JSONL data stays plain."""

GREEN = "\033[32m"
RESET = "\033[0m"


def fraction_text(numerator, denominator):
    rate = f"{numerator / denominator:.1%}" if denominator else "N/A"
    return f"{numerator}/{denominator} ({rate})"


def rate_text(value):
    return "N/A" if value is None else f"{value:.1%}"


def print_summary(summary, heading):
    print(heading, flush=True)
    attempts = retakes = defuses = excluded = 0
    time_expired = defender_eliminated = 0
    for opponent, left in summary["L"]["opponents"].items():
        right = summary["R"]["opponents"][opponent]
        played = left["rounds"]  # Both site summaries refer to the same rounds.
        count = left["retakes"] + right["retakes"]
        wins = left["defuses"] + right["defuses"]
        skipped = left["excluded_no_retake"]
        expired = left.get("time_expired", 0) + right.get("time_expired", 0)
        eliminated = left.get("defender_eliminated", 0) + right.get("defender_eliminated", 0)
        attempts += played
        retakes += count
        defuses += wins
        excluded += skipped
        time_expired += expired
        defender_eliminated += eliminated
        print(f"  {opponent}: defender wins {fraction_text(wins, count)} | losses={count - wins} "
              f"| L={left['defuses']}/{left['retakes']} R={right['defuses']}/{right['retakes']} "
              f"| time_expired={expired} defender_eliminated={eliminated}", flush=True)
        print(f"    {GREEN}retake_rate={fraction_text(count, played)} "
              f"| defuse_rate={fraction_text(wins, count)} | excluded={skipped}{RESET}", flush=True)
    print(f"  {GREEN}Overall: retake_rate={fraction_text(retakes, attempts)} "
          f"| defuse_rate={fraction_text(defuses, retakes)} | excluded={excluded} "
          f"| time_expired={time_expired} defender_eliminated={defender_eliminated}{RESET}", flush=True)
    for site in ("L", "R"):
        metric = summary[site]
        print(f"  {GREEN}{site} site: retakes={metric['retakes']} "
              f"| mean_defuse_rate={rate_text(metric['mean_defuse_rate'])} "
              f"| min_team_defuse_rate={rate_text(metric['min_defuse_rate'])}{RESET}", flush=True)
        print(f"    moving_fire={metric['moving_fire_rate']:.1%} "
              f"| smoke_defuse_decisions={metric['smoke_defuse_decisions']} "
              f"| time_expired={metric.get('time_expired', 0)} defender_eliminated={metric.get('defender_eliminated', 0)}", flush=True)


def print_evaluation_summary(summary):
    """Each model has its own attempts and ten-retake-per-opponent denominator."""
    _print_model_summary(summary, "evaluation")


def print_training_summary(summary, heading):
    print(heading, flush=True)
    _print_model_summary(summary, "training")


def _print_model_summary(summary, phase):
    for site in ("L", "R"):
        metric = summary[site]
        print(f"{site} model {phase}:", flush=True)
        total_rounds = total_defuses = total_excluded = 0
        for opponent, counts in metric["opponents"].items():
            retakes, wins = counts["retakes"], counts["defuses"]
            other = counts.get("excluded_other_site", 0)
            quota_full = counts.get("excluded_training_quota", 0)
            excluded = counts["excluded_no_retake"] + other + quota_full
            extra = f" quota_full={quota_full}" if phase == "training" else ""
            total_rounds += counts["rounds"]
            total_defuses += wins
            total_excluded += excluded
            print(f"  {opponent}: defender wins {fraction_text(wins, retakes)} | losses={retakes - wins} "
                  f"| time_expired={counts.get('time_expired', 0)} defender_eliminated={counts.get('defender_eliminated', 0)}", flush=True)
            print(f"    {GREEN}retake_rate={fraction_text(retakes, counts['rounds'])} "
                  f"| defuse_rate={fraction_text(wins, retakes)} | excluded={excluded} (other_site={other}{extra}){RESET}", flush=True)
        print(f"  {GREEN}{site} overall: retake_rate={fraction_text(metric['retakes'], total_rounds)} "
              f"| defuse_rate={fraction_text(total_defuses, metric['retakes'])} "
              f"| mean_defuse_rate={rate_text(metric['mean_defuse_rate'])} "
              f"| min_team_defuse_rate={rate_text(metric['min_defuse_rate'])} "
              f"| excluded={total_excluded}{RESET}", flush=True)
        print(f"    moving_fire={metric['moving_fire_rate']:.1%} "
              f"| smoke_defuse_decisions={metric['smoke_defuse_decisions']} "
              f"| time_expired={metric.get('time_expired', 0)} defender_eliminated={metric.get('defender_eliminated', 0)}", flush=True)


def print_retry_progress(opponent, completed, required, attempts, site=None):
    label = f"{site} model / {opponent}" if site is not None else opponent
    print(f"  {label}: waiting for retake | completed={completed}/{required} "
          f"| attempted_rounds={attempts}", flush=True)
    print(f"    {GREEN}retake_rate={fraction_text(completed, attempts)} "
          f"| excluded={attempts - completed}{RESET}", flush=True)


def format_elapsed_time(seconds):
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


