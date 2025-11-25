import time
import ast
from collections import Counter

import streamlit as st


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
        # Only numeric literals (int / float)
        value = node.n if isinstance(node, ast.Num) else node.value
        if not isinstance(value, (int, float)):
            raise ExpressionError("Only numeric literals are allowed.")

    # Disallow everything else
    elif isinstance(node, (ast.Call, ast.Name, ast.Subscript, ast.Attribute, ast.List, ast.Tuple)):
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

    # Ignore other nodes, since validate_ast already checked structure.


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


# ---------- Utility functions ----------

def parse_numbers_input(text: str):
    """
    Parse numbers from a string like '4 5 6 7 1 25' or '4,5,6,7,1,25'.
    Returns a list of ints.
    """
    if not text.strip():
        return []
    separators = [",", ";"]
    for sep in separators:
        text = text.replace(sep, " ")
    parts = text.split()
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            raise ValueError(f"'{p}' is not a valid integer.")
    return nums


def init_state():
    if "round_started" not in st.session_state:
        st.session_state.round_started = False
    if "numbers" not in st.session_state:
        st.session_state.numbers = []
    if "target" not in st.session_state:
        st.session_state.target = None
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "submissions" not in st.session_state:
        # For two players: index 0 and 1
        st.session_state.submissions = {
            0: None,
            1: None,
        }
    if "winner" not in st.session_state:
        st.session_state.winner = None


def start_round(numbers, target):
    st.session_state.numbers = numbers
    st.session_state.target = target
    st.session_state.round_started = True
    st.session_state.start_time = time.time()
    st.session_state.submissions = {0: None, 1: None}
    st.session_state.winner = None
    # Reset expression fields if they exist
    st.session_state["expr_0"] = ""
    st.session_state["expr_1"] = ""


def record_submission(player_index, name, expr):
    if not st.session_state.round_started:
        return

    if st.session_state.submissions[player_index] is not None:
        # Already submitted
        return

    submission_time = time.time()
    elapsed = submission_time - st.session_state.start_time

    numbers = st.session_state.numbers
    target = st.session_state.target

    status = {
        "name": name,
        "expression": expr,
        "elapsed": elapsed,
        "valid": False,
        "correct": False,
        "message": "",
        "result": None,
    }

    try:
        result, used_numbers = safe_evaluate_expression(expr)
        status["result"] = result

        # Check number usage
        ok_nums, msg_nums = check_number_usage(numbers, used_numbers)
        if not ok_nums:
            status["message"] = msg_nums
        else:
            # Check closeness to target
            if abs(result - target) < 1e-9:
                status["correct"] = True
                status["valid"] = True
                status["message"] = "Correct solution!"
            else:
                status["valid"] = True
                status["message"] = f"Expression evaluates to {result}, not {target}."

    except ExpressionError as e:
        status["message"] = f"Invalid expression: {e}"

    st.session_state.submissions[player_index] = status

    # Determine winner: first correct solution
    if status["correct"] and st.session_state.winner is None:
        st.session_state.winner = player_index


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="Numbers Tiebreak", layout="wide")
    init_state()

    st.title("Numbers Tiebreak Tool")

    # --- Sidebar: host controls ---
    st.sidebar.header("Host Controls")

    player1_name = st.sidebar.text_input("Player 1 name", "Player 1")
    player2_name = st.sidebar.text_input("Player 2 name", "Player 2")

    default_numbers = "4 5 6 7 1 25"
    numbers_text = st.sidebar.text_input(
        "Numbers (space/comma-separated)", default_numbers
    )
    target_value = st.sidebar.number_input("Target", value=106, step=1)

    if st.sidebar.button("Start round / reset"):
        try:
            numbers = parse_numbers_input(numbers_text)
            if not numbers:
                st.sidebar.error("You must provide at least one number.")
            else:
                start_round(numbers, int(target_value))
                st.sidebar.success("Round started!")
        except ValueError as e:
            st.sidebar.error(str(e))

    st.sidebar.markdown("---")
    st.sidebar.write("Rules:")
    st.sidebar.write("- Use only the given numbers (each at most once).")
    st.sidebar.write("- Use only +, -, *, / and parentheses.")
    st.sidebar.write("- Enter a single expression that equals the target.")

    # --- Main area ---

    if not st.session_state.round_started:
        st.info("Set the numbers and target in the sidebar, then click **Start round / reset**.")
        return

    # Display current puzzle
    st.subheader("Current Round")

    nums_display = " ".join(str(n) for n in st.session_state.numbers)
    st.markdown(f"**Numbers:** {nums_display}")
    st.markdown(f"**Target:** {st.session_state.target}")

    # Two columns for the two players
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {player1_name}")
        expr1 = st.text_area(
            "Enter your expression:",
            key="expr_0",
            height=150,
            placeholder="Example: 4*25+6",
        )
        if st.button("I'm done!", key="done_0"):
            record_submission(0, player1_name, expr1)

        sub1 = st.session_state.submissions[0]
        if sub1 is not None:
            st.markdown("---")
            st.markdown(f"**Time:** {sub1['elapsed']:.2f} seconds")
            st.markdown(f"**Expression:** `{sub1['expression']}`")
            if sub1["result"] is not None:
                st.markdown(f"**Value:** {sub1['result']}")
            st.markdown(f"**Status:** {sub1['message']}")

    with col2:
        st.markdown(f"### {player2_name}")
        expr2 = st.text_area(
            "Enter your expression:",
            key="expr_1",
            height=150,
            placeholder="Example: 4*25+6",
        )
        if st.button("I'm done!", key="done_1"):
            record_submission(1, player2_name, expr2)

        sub2 = st.session_state.submissions[1]
        if sub2 is not None:
            st.markdown("---")
            st.markdown(f"**Time:** {sub2['elapsed']:.2f} seconds")
            st.markdown(f"**Expression:** `{sub2['expression']}`")
            if sub2["result"] is not None:
                st.markdown(f"**Value:** {sub2['result']}")
            st.markdown(f"**Status:** {sub2['message']}")

    # --- Results summary ---
    st.markdown("---")
    st.subheader("Result")

    winner_idx = st.session_state.winner
    sub1 = st.session_state.submissions[0]
    sub2 = st.session_state.submissions[1]

    if winner_idx is None:
        # No correct solution yet
        if (sub1 and sub1["valid"]) or (sub2 and sub2["valid"]):
            st.write("No correct solution submitted yet.")
        else:
            st.write("Waiting for submissions...")
    else:
        winner_name = player1_name if winner_idx == 0 else player2_name
        winner_sub = st.session_state.submissions[winner_idx]
        st.success(
            f"🏆 **{winner_name}** wins with a correct solution "
            f"in {winner_sub['elapsed']:.2f} seconds!"
        )


if __name__ == "__main__":
    main()
