import time
import ast
from collections import Counter

import streamlit as st
from streamlit_autorefresh import st_autorefresh


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
    game["submissions"] = {"p1": None, "p2": None}
    game["winner"] = None


def reset_game(game):
    """Reset the game state (but keep player names)."""
    game["numbers"] = []
    game["target"] = None
    game["round_started"] = False
    game["start_time"] = None
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

        # Distance from target based on declared answer
        distance = abs(sub["declared"] - target)
        sub["distance"] = distance  # store for display

        if best is None:
            best = (distance, sub["elapsed"], player_key)
        else:
            if distance < best[0] or (distance == best[0] and sub["elapsed"] < best[1]):
                best = (distance, sub["elapsed"], player_key)

    game["winner"] = best[2] if best else None


def record_submission(game, player_key, declared_value, expr):
    if not game["round_started"]:
        return

    if game["submissions"][player_key] is not None:
        # Already submitted
        return

    submission_time = time.time()
    elapsed = submission_time - game["start_time"]

    numbers = game["numbers"]
    target = game["target"]
    name = game["player_names"][player_key]

    status = {
        "name": name,
        "declared": declared_value,
        "expression": expr,
        "elapsed": elapsed,
        "valid": False,
        "message": "",
        "result": None,
        "distance": None,
        "numbers_ok": False,
        "expr_matches_declared": False,
    }

    try:
        result, used_numbers = safe_evaluate_expression(expr)
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
    # True auto-refresh from frontend: every 700ms
    st_autorefresh(interval=700, key=f"host_refresh_{game_id}")

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

    # Optional: show a brief "GO" message for host as well
    if game["start_time"] is not None:
        elapsed_since_start = time.time() - game["start_time"]
        if elapsed_since_start < 3:
            st.success("🚦 Round started! Contestants, GO!")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {game['player_names']['p1']}")
        sub1 = game["submissions"]["p1"]
        if sub1 is None:
            st.write("No submission yet.")
        else:
            st.markdown(f"- **Time:** {sub1['elapsed']:.2f} seconds")
            st.markdown(f"- **Declared:** {sub1['declared']}")
            st.markdown(f"- **Expression:** `{sub1['expression']}`")
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
            st.markdown(f"- **Expression:** `{sub2['expression']}`")
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
            f"(declared {winner_sub['declared']}, distance {winner_sub['distance']}, "
            f"time {winner_sub['elapsed']:.2f}s)"
        )


# ---------- UI: Player view ----------

def player_view(game_id: str, player_key: str):
    # True auto-refresh from frontend: every 700ms
    st_autorefresh(interval=700, key=f"{player_key}_refresh_{game_id}")

    game = get_game(game_id)
    name = game["player_names"][player_key]

    st.title(f"Numbers Tiebreak – {name}")

    declared_key = f"declared_{player_key}"
    expr_key = f"expr_{player_key}"

    # If round not started, clear any previous expression/declared and show waiting
    if not game["round_started"]:
        if declared_key in st.session_state:
            st.session_state[declared_key] = 0
        if expr_key in st.session_state:
            st.session_state[expr_key] = ""
        st.info("The host has not started the round yet. Please wait...")
        return

    nums_display = " ".join(str(n) for n in game["numbers"])
    st.subheader(f"Game: {game_id}")
    st.markdown(f"**Numbers:** {nums_display}")
    st.markdown(f"**Target:** {game['target']}")

    # Flash a big START banner for the first 3 seconds
    if game["start_time"] is not None:
        elapsed_since_start = time.time() - game["start_time"]
        if elapsed_since_start < 3:
            st.success("🚦 Round started! GO!")

    st.markdown("---")

    declared = st.number_input(
        "Your declared result:",
        key=declared_key,
        step=1,
        format="%d",
    )

    expr = st.text_area(
        "Your method (expression):",
        key=expr_key,
        height=150,
        placeholder="Example: 4*25+6",
    )

    if st.button("I'm done!"):
        # declared is a number; we pass it as int
        record_submission(game, player_key, int(declared), expr)

    submission = game["submissions"][player_key]
    if submission is not None:
        st.markdown("---")
        st.markdown(f"**Time:** {submission['elapsed']:.2f} seconds")
        st.markdown(f"**Declared:** {submission['declared']}")
        st.markdown(f"**Expression:** `{submission['expression']}`")
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
