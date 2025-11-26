import time
import ast
from collections import Counter

import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ---------- Config ----------

ROUND_TIME_LIMIT = 30  # seconds


# ---------- Expression handling ----------

class ExpressionError(Exception):
    """Custom exception for expression errors."""
    pass


def validate_ast(node):
    """Recursively validate that the AST uses only allowed operations."""
    if isinstance(node, ast.Expression):
        validate_ast(node.body)

    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise ExpressionError("Only +, -, *, and / are allowed.")
        validate_ast(node.left)
        validate_ast(node.right)

    elif isinstance(node, ast.UnaryOp):
        # Allow unary + and -
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ExpressionError("Only +, -, *, and / are allowed.")
        validate_ast(node.operand)

    elif isinstance(node, (ast.Constant, ast.Num)):
        value = node.n if isinstance(node, ast.Num) else node.value
        if not isinstance(value, (int, float)):
            raise ExpressionError("Only numeric literals are allowed.")

    elif isinstance(node, (ast.Call, ast.Name, ast.Subscript,
                           ast.Attribute, ast.List, ast.Tuple)):
        raise ExpressionError("Only numbers, parentheses and +, -, *, / are allowed.")
    else:
        raise ExpressionError("Invalid expression structure.")


def collect_numbers(node, acc):
    """Collect all numeric literals from the AST into acc list."""
    if isinstance(node, ast.Expression):
        collect_numbers(node.body, acc)

    elif isinstance(node, ast.BinOp):
        collect_numbers(node.left, acc)
        collect_numbers(node.right, acc)

    elif isinstance(node, ast.UnaryOp):
        collect_numbers(node.operand, acc)

    elif isinstance(node, (ast.Constant, ast.Num)):
        value = node.n if isinstance(node, ast.Num) else node.value
        if isinstance(value, (int, float)):
            acc.append(int(value))


def safe_evaluate_expression(expr: str):
    """
    Parse and safely evaluate an arithmetic expression using only +, -, *, /.
    Returns (result, used_numbers) where used_numbers is a list of ints.
    Raises ExpressionError on failure.
    """
    expr = expr.strip()
    if not expr:
        raise ExpressionError("Expression is empty.")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ExpressionError("Could not parse expression (syntax error).")

    validate_ast(tree)
    used_numbers = []
    collect_numbers(tree, used_numbers)

    try:
        compiled = compile(tree, "<expr>", "eval")
        result = eval(compiled, {"__builtins__": None}, {})
    except Exception:
        raise ExpressionError("Error while evaluating expression.")

    return result, used_numbers


def check_number_usage(available_numbers, used_numbers):
    """
    Ensure each literal is used at most as many times as available.
    Returns (ok, message).
    """
    available_counts = Counter(available_numbers)
    used_counts = Counter(used_numbers)

    for num, count in used_counts.items():
        if count > available_counts.get(num, 0):
            return (
                False,
                f"Number {num} used {count} times but only {available_counts.get(num, 0)} available.",
            )

    return True, "Number usage is valid."


# ---------- Shared game store (across sessions) ----------

@st.cache_resource
def get_store():
    """
    Returns a singleton dict shared by all users/sessions.
    Structure: { game_id: game_dict, ... }
    """
    return {}


def get_game(game_id: str):
    store = get_store()
    if game_id not in store:
        store[game_id] = {
            "numbers": [],
            "target": None,
            "round_started": False,
            "start_time": None,
            "stop_time": None,          # when host stops timer early
            "player_names": {"p1": "Player 1", "p2": "Player 2"},
            "submissions": {"p1": None, "p2": None},
            "winner": None,
        }
    return store[game_id]


# ---------- Helpers ----------

def parse_numbers_input(text: str):
    """
    Parse numbers from a string like '4 5 6 7 1 25' or '4,5,6,7,1,25'.
    Returns a list of ints.
    """
    if not text.strip():
        return []
    for sep in [",", ";"]:
        text = text.replace(sep, " ")
    parts = text.split()
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            raise ValueError(f"'{p}' is not a valid integer.")
    return nums


def start_round(game, numbers, target):
    game["numbers"] = numbers
    game["target"] = target
    game["round_started"] = True
    game["start_time"] = time.time()
    game["stop_time"] = None
    game["submissions"] = {"p1": None, "p2": None}
    game["winner"] = None


def reset_game(game):
    """Reset the game state (but keep player names)."""
    game["numbers"] = []
    game["target"] = None
    game["round_started"] = False
    game["start_time"] = None
    game["stop_time"] = None
    game["submissions"] = {"p1": None, "p2": None}
    game["winner"] = None


def recompute_winner(game):
    """Apply 'closest to target wins, time as tiebreak' among valid submissions."""
    target = game["target"]
    best = None  # (distance, elapsed, player_key)

    for player_key in ("p1", "p2"):
        sub = game["submissions"].get(player_key)
        if not sub:
            continue
        if not sub.get("valid", False):
            continue

        distance = abs(sub["declared"] - target)
        sub["distance"] = distance  # store for display

        if best is None:
            best = (distance, sub["elapsed"], player_key)
        else:
            if distance < best[0] or (distance == best[0] and sub["elapsed"] < best[1]):
                best = (distance, sub["elapsed"], player_key)

    game["winner"] = best[2] if best else None


def sanitise_expression(expr: str) -> str:
    """Allow 'x'/'X'/ '×' as multiplication by converting to '*' internally."""
    return (
        expr.replace("×", "*")
        .replace("x", "*")
        .replace("X", "*")
    )


def record_submission(game, player_key, declared_value, expr, allow_after_end: bool = False):
    """
    Record a submission.

    - If allow_after_end=False: submissions after cutoff are rejected as 'too late'.
    - If allow_after_end=True: used for auto-submit at round end; we accept and clamp
      elapsed to the cutoff time.
    """
    if not game["round_started"]:
        return

    # Determine effective cutoff (natural timeout or host stop)
    if game["stop_time"] is not None:
        cutoff_elapsed = game["stop_time"] - game["start_time"]
    else:
        cutoff_elapsed = ROUND_TIME_LIMIT

    elapsed_from_start = time.time() - game["start_time"]

    if (not allow_after_end) and elapsed_from_start > cutoff_elapsed:
        late_reason = (
            "round was stopped by the host."
            if game["stop_time"] is not None
            else f">{ROUND_TIME_LIMIT}s time limit."
        )
        game["submissions"][player_key] = {
            "name": game["player_names"][player_key],
            "declared": declared_value,
            "expression": expr,
            "expression_eval": None,
            "elapsed": elapsed_from_start,
            "valid": False,
            "message": f"Submission too late ({late_reason})",
            "result": None,
            "distance": None,
            "numbers_ok": False,
            "expr_matches_declared": False,
        }
        recompute_winner(game)
        return

    if game["submissions"][player_key] is not None:
        # Already submitted
        return

    # If we are auto-submitting after cutoff, clamp elapsed to cutoff
    if allow_after_end and elapsed_from_start > cutoff_elapsed:
        elapsed_from_start = cutoff_elapsed

    numbers = game["numbers"]
    target = game["target"]
    name = game["player_names"][player_key]

    expr_for_eval = sanitise_expression(expr)

    status = {
        "name": name,
        "declared": declared_value,
        "expression": expr,          # what they typed
        "expression_eval": expr_for_eval,  # what we actually evaluated
        "elapsed": elapsed_from_start,
        "valid": False,
        "message": "",
        "result": None,
        "distance": None,
        "numbers_ok": False,
        "expr_matches_declared": False,
    }

    try:
        result, used_numbers = safe_evaluate_expression(expr_for_eval)
        status["result"] = result

        # Check number usage
        ok_nums, msg_nums = check_number_usage(numbers, used_numbers)
        status["numbers_ok"] = ok_nums
        if not ok_nums:
            status["message"] = msg_nums
        else:
            # Check expression vs declared answer
            if declared_value is None:
                status["message"] = "You must enter a declared answer."
            else:
                if abs(result - declared_value) < 1e-9:
                    status["expr_matches_declared"] = True
                    status["valid"] = True
                    status["distance"] = abs(declared_value - target)
                    status["message"] = (
                        f"Valid submission. Declared {declared_value}, "
                        f"distance {status['distance']} from target."
                    )
                else:
                    status["message"] = (
                        f"Expression evaluates to {result}, "
                        f"which does not match your declared answer {declared_value}."
                    )

    except ExpressionError as e:
        status["message"] = f"Invalid expression: {e}"

    game["submissions"][player_key] = status

    # Recompute winner based on all valid submissions so far
    recompute_winner(game)


# ---------- UI: Host view ----------

def host_view(game_id: str):
    # True auto-refresh from frontend: every 1000ms
    st_autorefresh(interval=1000, key=f"host_refresh_{game_id}")

    game = get_game(game_id)

    st.title("Numbers Tiebreak – Host")

    st.sidebar.header("Game selection")
    st.sidebar.write("Game ID (used in player links):")
    st.sidebar.code(game_id)

    st.sidebar.header("Player names")
    p1_name = st.sidebar.text_input("Player 1 name", game["player_names"]["p1"])
    p2_name = st.sidebar.text_input("Player 2 name", game["player_names"]["p2"])
    game["player_names"]["p1"] = p1_name
    game["player_names"]["p2"] = p2_name

    st.sidebar.header("Round setup")

    default_numbers = "4 5 6 7 1 25" if not game["numbers"] else " ".join(
        str(n) for n in game["numbers"]
    )
    numbers_text = st.sidebar.text_input(
        "Numbers (space/comma-separated)", default_numbers
    )

    default_target = game["target"] if game["target"] is not None else 106
    target_value = st.sidebar.number_input("Target", value=int(default_target), step=1)

    start_clicked = st.sidebar.button("▶ Start round")
    reset_clicked = st.sidebar.button("⏹ Reset game")

    if start_clicked:
        try:
            numbers = parse_numbers_input(numbers_text)
            if not numbers:
                st.sidebar.error("You must provide at least one number.")
            else:
                start_round(game, numbers, int(target_value))
                st.sidebar.success("Round started!")
        except ValueError as e:
            st.sidebar.error(str(e))

    if reset_clicked:
        reset_game(game)
        st.sidebar.warning("Game reset.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Player links")
    st.sidebar.write("Share these *suffixes* and add them to your deployed app URL:")
    st.sidebar.code(f"?role=p1&game={game_id}", language="text")
    st.sidebar.code(f"?role=p2&game={game_id}", language="text")

    # Main content
    st.subheader(f"Game: {game_id}")

    if not game["round_started"]:
        st.info("Round not started. Set numbers & target, then click **Start round**.")
        return

    nums_display = " ".join(str(n) for n in game["numbers"])
    st.markdown(f"**Numbers:** {nums_display}")
    st.markdown(f"**Target:** {game['target']}")
    st.markdown(f"**Time limit:** {ROUND_TIME_LIMIT} seconds")

    # Timer / progress bar + Stop Timer & solution link
    if game["start_time"] is not None:
        if game["stop_time"] is not None:
            raw_elapsed = game["stop_time"] - game["start_time"]
        else:
            raw_elapsed = time.time() - game["start_time"]

        elapsed_display = min(ROUND_TIME_LIMIT, raw_elapsed)
        round_ended_naturally = raw_elapsed >= ROUND_TIME_LIMIT
        round_ended = (game["stop_time"] is not None) or round_ended_naturally
        remaining_display = 0 if round_ended else max(0, ROUND_TIME_LIMIT - raw_elapsed)
        progress_pct = int((elapsed_display / ROUND_TIME_LIMIT) * 100)

        col_timer, col_button = st.columns([3, 1])

        with col_timer:
            st.markdown(
                f"**Time elapsed:** {elapsed_display:.0f}s  •  "
                f"**Remaining:** {remaining_display:.0f}s"
            )
            st.progress(progress_pct)

        with col_button:
            if not round_ended:
                if st.button("⏹ Stop timer now"):
                    game["stop_time"] = time.time()
                    round_ended = True

        if not round_ended:
            if raw_elapsed < 3:
                st.success("🚦 Round started! Contestants, GO!")
        else:
            st.error("⏰ Round ended.")
            # Link to external solution page
            base_url = "https://greem.co.uk/quantumtombola/"
            sel_param = "-".join(str(n) for n in game["numbers"])
            target_param = game["target"]
            solution_url = f"{base_url}?sel={sel_param}&target={target_param}"
            st.markdown(
                f"[🔍 View solution page]({solution_url})",
                unsafe_allow_html=False,
            )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {game['player_names']['p1']}")
        sub1 = game["submissions"]["p1"]
        if sub1 is None:
            st.write("No submission yet.")
        else:
            st.markdown(f"- **Time:** {sub1['elapsed']:.2f} seconds")
            st.markdown(f"- **Declared:** {sub1['declared']}")
            st.markdown(f"- **Expression (typed):** `{sub1['expression']}`")
            if sub1["expression_eval"] != sub1["expression"]:
                st.markdown(f"- **Expression (interpreted):** `{sub1['expression_eval']}`")
            if sub1["result"] is not None:
                st.markdown(f"- **Expression value:** {sub1['result']}")
            if sub1["distance"] is not None:
                st.markdown(f"- **Distance from target:** {sub1['distance']}")
            st.markdown(f"- **Status:** {sub1['message']}")

    with col2:
        st.markdown(f"### {game['player_names']['p2']}")
        sub2 = game["submissions"]["p2"]
        if sub2 is None:
            st.write("No submission yet.")
        else:
            st.markdown(f"- **Time:** {sub2['elapsed']:.2f} seconds")
            st.markdown(f"- **Declared:** {sub2['declared']}")
            st.markdown(f"- **Expression (typed):** `{sub2['expression']}`")
            if sub2["expression_eval"] != sub2["expression"]:
                st.markdown(f"- **Expression (interpreted):** `{sub2['expression_eval']}`")
            if sub2["result"] is not None:
                st.markdown(f"- **Expression value:** {sub2['result']}")
            if sub2["distance"] is not None:
                st.markdown(f"- **Distance from target:** {sub2['distance']}")
            st.markdown(f"- **Status:** {sub2['message']}")

    st.markdown("---")
    st.subheader("Result")

    winner = game["winner"]

    if winner is None:
        st.write("No valid winning submission yet.")
    else:
        winner_name = game["player_names"][winner]
        winner_sub = game["submissions"][winner]
        st.success(
            f"🏆 **{winner_name}** wins! "
            f"(declared {winner_sub['declared']}, "
            f"distance {winner_sub['distance']}, "
            f"time {winner_sub['elapsed']:.2f}s)"
        )


# ---------- UI: Player view ----------

def player_view(game_id: str, player_key: str):
    # True auto-refresh from frontend: every 1000ms
    st_autorefresh(interval=1000, key=f"{player_key}_refresh_{game_id}")

    game = get_game(game_id)
    name = game["player_names"][player_key]

    st.title(f"Numbers Tiebreak – {name}")

    declared_key = f"declared_{player_key}"
    expr_key = f"expr_{player_key}"

    # If round not started, clear any previous expression/declared and show waiting
    if not game["round_started"]:
        st.session_state.pop(declared_key, None)
        st.session_state.pop(expr_key, None)
        st.info("The host has not started the round yet. Please wait...")
        return

    nums_display = " ".join(str(n) for n in game["numbers"])
    st.subheader(f"Game: {game_id}")
    st.markdown(f"**Numbers:** {nums_display}")
    st.markdown(f"**Target:** {game['target']}")
    st.markdown(f"**Time limit:** {ROUND_TIME_LIMIT} seconds")

    # Timer / progress bar respecting stop_time
    if game["stop_time"] is not None:
        raw_elapsed = game["stop_time"] - game["start_time"]
        round_ended = True
    else:
        raw_elapsed = time.time() - game["start_time"]
        round_ended = raw_elapsed >= ROUND_TIME_LIMIT

    elapsed_clamped = min(ROUND_TIME_LIMIT, raw_elapsed)
    remaining = 0 if round_ended else max(0, ROUND_TIME_LIMIT - raw_elapsed)
    progress_pct = int((elapsed_clamped / ROUND_TIME_LIMIT) * 100)

    st.markdown(f"**Time remaining:** {remaining:.0f} seconds")
    st.progress(progress_pct)

    if not round_ended and raw_elapsed < 3:
        st.success("🚦 Round started! GO!")
    elif round_ended:
        st.error("⏰ Round has ended.")

    st.markdown("---")

    # Default declared value = target, unless user already typed something
    if declared_key not in st.session_state:
        st.session_state[declared_key] = game["target"]

    # Get current widget values (or defaults)
    declared = st.number_input(
        "Your declared result:",
        key=declared_key,
        step=1,
        format="%d",
        disabled=False,  # we'll disable logically below
    )

    expr = st.text_area(
        "Your method (expression):",
        key=expr_key,
        height=150,
        placeholder="Example: 4*25+6 or 4x25+6",
        disabled=False,
    )

    submission = game["submissions"][player_key]
    has_submitted = submission is not None

    # If round just ended and this player hasn't submitted yet,
    # auto-submit whatever is currently in the boxes.
    if round_ended and not has_submitted:
        record_submission(game, player_key, int(declared), expr, allow_after_end=True)
        submission = game["submissions"][player_key]
        has_submitted = True

    # Recompute can_submit / disabled flags now that auto-submit may have occurred
    can_submit = (not round_ended) and (not has_submitted)

    # Show a strong banner after submission
    if has_submitted:
        if submission["valid"]:
            st.success("✅ Submission received and recorded. Your method matches your declaration.")
        else:
            st.warning("⚠️ Submission received, but it's invalid. Check the status details below.")

    # Colored panel around the inputs (green/red/neutral)
    if has_submitted:
        if submission["valid"]:
            panel_color = "#dcfce7"  # light green
            border_color = "#16a34a"
        else:
            panel_color = "#fef3c7"  # light amber
            border_color = "#f59e0b"
    else:
        panel_color = "#f1f5f9"  # neutral
        border_color = "#cbd5f5"

    # Re-render the inputs inside a coloured wrapper but keep them disabled
    st.markdown(
        f"""
        <div style="
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            border: 1px solid {border_color};
            background-color: {panel_color};
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        ">
        """,
        unsafe_allow_html=True,
    )

    # Disabled copies (to prevent further edits after end/submission)
    st.number_input(
        "Your declared result:",
        value=int(declared),
        step=1,
        format="%d",
        key=f"{declared_key}_display",
        disabled=True,
    )

    st.text_area(
        "Your method (expression):",
        value=expr,
        height=150,
        key=f"{expr_key}_display",
        disabled=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if can_submit:
        if st.button("I'm done!"):
            record_submission(game, player_key, int(declared), expr)
    else:
        if not has_submitted:
            st.write("Submission disabled: round has ended.")
        else:
            st.write("You have already submitted.")

    # Show submission details
    submission = game["submissions"][player_key]
    if submission is not None:
        st.markdown("---")
        st.markdown(f"**Time:** {submission['elapsed']:.2f} seconds")
        st.markdown(f"**Declared:** {submission['declared']}")
        st.markdown(f"**Expression (typed):** `{submission['expression']}`")
        if submission["expression_eval"] != submission["expression"]:
            st.markdown(f"**Expression (interpreted):** `{submission['expression_eval']}`")
        if submission["result"] is not None:
            st.markdown(f"**Expression value:** {submission['result']}")
        if submission["distance"] is not None:
            st.markdown(f"**Distance from target:** {submission['distance']}")
        st.markdown(f"**Status:** {submission['message']}")


# ---------- Main router ----------

def main():
    st.set_page_config(page_title="Numbers Tiebreak", layout="wide")

    # Read query params to determine role and game
    try:
        params = st.query_params  # new API
    except AttributeError:
        params = st.experimental_get_query_params()  # fallback

    # Normalize to strings
    role = params.get("role", "host")
    if isinstance(role, list):
        role = role[0]

    game_id = params.get("game", "default")
    if isinstance(game_id, list):
        game_id = game_id[0]

    if role not in {"host", "p1", "p2"}:
        role = "host"

    if role == "host":
        host_view(game_id)
    elif role == "p1":
        player_view(game_id, "p1")
    elif role == "p2":
        player_view(game_id, "p2")


if __name__ == "__main__":
    main()
