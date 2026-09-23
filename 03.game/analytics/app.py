from flask import Flask

if __package__:
    from .analysis_branching import AI_ANALYSIS_RULES, analyze_match_data
    from .match_calculation import calculate_match_data_from_original
    from .match_output import (
        get_all_matches,
        index,
        match_detail,
        save_training_labels,
        save_favorite,
    )
else:
    from analysis_branching import AI_ANALYSIS_RULES, analyze_match_data
    from match_calculation import calculate_match_data_from_original
    from match_output import (
        get_all_matches,
        index,
        match_detail,
        save_training_labels,
        save_favorite,
    )

app = Flask(__name__)
app.add_url_rule("/match/<path:match_path>", view_func=match_detail)
app.add_url_rule(
    "/match/<path:match_path>/label",
    view_func=save_training_labels,
    methods=["POST"],
)
app.add_url_rule(
    "/match/<path:match_path>/favorite",
    view_func=save_favorite,
    methods=["POST"],
)
app.add_url_rule("/", view_func=index)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
