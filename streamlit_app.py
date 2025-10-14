import streamlit as st
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Dict

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="Tic-Tac-Toe Games", page_icon="🕹️")

st.markdown("""
    <style>
    .stApp {
        background-color: #4A4548;
    }
            
    .stButton>button {
        border-radius: 10px;
        background-color: #BA4C4C;
        font-weight: 600;
    }

    div[data-testid="stMetricLabel"] {
    color: white !important;
    font-weight: 600;
    font-size: 1rem;
    }

    div[data-testid="stMetricValue"] {
    color: E7C5C5 !important;
    font-weight: 700;
    font-size: 1.5rem;
    }

    div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 10px;
    border-radius: 8px;
    }

    .white-text {
    color: white;
    font-size: 0.9rem; 
    }

    [data-testid="stSidebarContent"] {
        background-color: #292528; 
        color: #FFFFFF; 
    }
            
    div[data-testid="stButton"] button {
        aspect-ratio: 1 / 1;
        height: auto;       
        font-size: 2rem;         
    }
            
    div.stButton > button:first-child {
    color: #E7C5C5;
    }

    .stButton > button:hover {
    background-color: #C5E7E7;
    transform: scale(1.02);
    }
            
    .header {
    color: #FFFFFF;
    text-align: center;
    padding: 2.5rem 1rem;
    }

    section[data-testid="stSidebar"] label p {
    color: white !important;
    font-weight: bold;     /* optional */
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: white !important;
    background-color: #222 !important;  /* optional: dark background for contrast */
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] {
    border: 1px solid #555 !important;
    border-radius: 6px;
    }

    </style>
""", unsafe_allow_html=True)

# ---------------------- Game & AI Core ----------------------
WIN_COMBOS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6)               # diagonals
]

def check_winner(board: List[str]) -> Tuple[Optional[str], Optional[Tuple[int,int,int]]]:
    for a,b,c in WIN_COMBOS:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a], (a,b,c)
    if all(board):
        return "Draw", None
    return None, None

def available_moves(board: List[str]) -> List[int]:
    return [i for i, v in enumerate(board) if v == ""]

def evaluate(board: List[str], ai_symbol: str, depth: int) -> int:
    # +10 (AI wins), -10 (opponent wins), 0 draw
    winner, _line = check_winner(board)
    opp = "O" if ai_symbol == "X" else "X"
    if winner == ai_symbol:
        return 10 - depth  # prefer quicker wins
    if winner == opp:
        return -10 + depth # prefer slower losses
    if winner == "Draw":
        return 0
    return None  # non-terminal

@dataclass
class SearchStats:
    nodes: int = 0
    pruned: int = 0
    decision_time_s: float = 0.0

def minimax_decide(board: List[str], player: str, ai_symbol: str) -> Tuple[int, SearchStats]:
    stats = SearchStats()
    def minimax(state: List[str], current: str, depth: int) -> int:
        stats.nodes += 1
        score = evaluate(state, ai_symbol, depth)
        if score is not None:
            return score

        is_max = (current == ai_symbol)
        moves = available_moves(state)
        if is_max:
            best = -10_000
            for m in moves:
                state[m] = current
                best = max(best, minimax(state, "O" if current == "X" else "X", depth+1))
                state[m] = ""
            return best
        else:
            best = 10_000
            for m in moves:
                state[m] = current
                best = min(best, minimax(state, "O" if current == "X" else "X", depth+1))
                state[m] = ""
            return best

    t0 = time.perf_counter()
    best_score = -10_000
    best_move = None
    for m in available_moves(board):
        board[m] = player
        score = minimax(board, "O" if player == "X" else "X", 1)
        board[m] = ""
        if score > best_score:
            best_score = score
            best_move = m
    stats.decision_time_s = time.perf_counter() - t0
    return best_move if best_move is not None else random.choice(available_moves(board)), stats

def alphabeta_decide(board: List[str], player: str, ai_symbol: str) -> Tuple[int, SearchStats]:
    stats = SearchStats()
    def alphabeta(state: List[str], current: str, depth: int, alpha: int, beta: int) -> int:
        stats.nodes += 1
        score = evaluate(state, ai_symbol, depth)
        if score is not None:
            return score

        is_max = (current == ai_symbol)
        moves = available_moves(state)

        if is_max:
            value = -10_000
            for m in moves:
                state[m] = current
                value = max(value, alphabeta(state, "O" if current == "X" else "X", depth+1, alpha, beta))
                state[m] = ""
                alpha = max(alpha, value)
                if alpha >= beta:
                    # prune remaining moves
                    stats.pruned += len(moves) - (moves.index(m) + 1)
                    break
            return value
        else:
            value = 10_000
            for m in moves:
                state[m] = current
                value = min(value, alphabeta(state, "O" if current == "X" else "X", depth+1, alpha, beta))
                state[m] = ""
                beta = min(beta, value)
                if alpha >= beta:
                    stats.pruned += len(moves) - (moves.index(m) + 1)
                    break
            return value

    t0 = time.perf_counter()
    best_score = -10_000
    best_move = None
    alpha, beta = -10_000, 10_000
    for m in available_moves(board):
        board[m] = player
        score = alphabeta(board, "O" if player == "X" else "X", 1, alpha, beta)
        board[m] = ""
        if score > best_score:
            best_score = score
            best_move = m
        alpha = max(alpha, best_score)
    stats.decision_time_s = time.perf_counter() - t0
    return best_move if best_move is not None else random.choice(available_moves(board)), stats

# ---------------------- Session State ----------------------
def init_state():
    if "board" not in st.session_state:
        st.session_state.board = [""] * 9
    if "current" not in st.session_state:
        st.session_state.current = "X"
    if "winner" not in st.session_state:
        st.session_state.winner = None
    if "win_line" not in st.session_state:
        st.session_state.win_line = None
    if "mode" not in st.session_state:
        st.session_state.mode = "Human vs AI"
    if "human_symbol" not in st.session_state:
        st.session_state.human_symbol = "X"
    if "ai1_algo" not in st.session_state:
        st.session_state.ai1_algo = "Alpha-Beta"
    if "ai2_algo" not in st.session_state:
        st.session_state.ai2_algo = "Minimax"
    if "perf_log" not in st.session_state:
        st.session_state.perf_log = []  # list of dicts per AI move
    if "autoplay" not in st.session_state:
        st.session_state.autoplay = False
    if "ai_delay" not in st.session_state:
        st.session_state.ai_delay = 0.5

init_state()

# ---------------------- Sidebar Controls ----------------------
st.sidebar.header("Game Settings")
st.sidebar.caption("Press **Restart** to clean moves made")

st.session_state.mode = st.sidebar.selectbox("Mode Selection", ["Human vs Human", "Human vs AI", "AI vs AI (Auto-play)"])

if st.session_state.mode == "Human vs AI":
    st.session_state.human_symbol = st.sidebar.radio("You play as", ["X", "O"], horizontal=True)
    st.session_state.ai1_algo = st.sidebar.radio("AI Algorithm", ["Minimax", "Alpha-Beta"], index=1, horizontal=True)
elif st.session_state.mode == "AI vs AI (Auto-play)":
    st.session_state.ai1_algo = st.sidebar.selectbox("AI for X", ["Minimax", "Alpha-Beta"], index=1)
    st.session_state.ai2_algo = st.sidebar.selectbox("AI for O", ["Minimax", "Alpha-Beta"], index=0)
    st.session_state.ai_delay = st.sidebar.slider("Auto-play speed (sec per move)", 0.0, 2.0, st.session_state.ai_delay, 0.1)
else:
    # Human vs Human
    pass

colA, colB = st.sidebar.columns(2)
if colA.button("Restart"):
    st.session_state.board = [""]*9
    st.session_state.current = "X"
    st.session_state.winner = None
    st.session_state.win_line = None
    st.session_state.perf_log = []
    st.session_state.autoplay = False
if st.session_state.mode == "AI vs AI (Auto-play)":
    play_col1, play_col2 = st.sidebar.columns(2)
    if play_col1.button("▶ Next"):
        st.session_state.autoplay = True
    if play_col2.button("⏸ Stop"):
        st.session_state.autoplay = False

# ---------------------- Helpers ----------------------
def algo_fn(name: str) -> Callable[[List[str], str, str], Tuple[int, SearchStats]]:
    return minimax_decide if name == "Minimax" else alphabeta_decide

def drop_piece(idx: int, symbol: str):
    if st.session_state.board[idx] == "" and not st.session_state.winner:
        st.session_state.board[idx] = symbol
        st.session_state.current = "O" if symbol == "X" else "X"
        w, line = check_winner(st.session_state.board)
        st.session_state.winner = w
        st.session_state.win_line = line

def ai_move(which_algo: str, ai_symbol: str):
    # Decide and play; collect metrics
    decide = algo_fn(which_algo)
    placeholder = st.empty()
    with st.spinner(f"AI ({which_algo}) is thinking..."):
        move, stats = decide(st.session_state.board.copy(), st.session_state.current, ai_symbol)
    # Log metrics
    pruned_pct = (stats.pruned / stats.nodes * 100.0) if stats.nodes > 0 else 0.0
    st.session_state.perf_log.append({
        "move_number": len([b for b in st.session_state.board if b != ""]) + 1,
        "player": st.session_state.current,
        "algorithm": which_algo,
        "decision_time_ms": round(stats.decision_time_s * 1000.0, 3),
        "nodes_explored": stats.nodes,
        "nodes_pruned": stats.pruned,
        "prune_efficiency_%": round(pruned_pct, 2),
        "chosen_move": move
    })
    drop_piece(move, st.session_state.current)

# ---------------------- Header ----------------------
st.markdown('<div class="header"><h1>Tic-Tac-Toe Games</h1><p>Double click a tile to make a move</p></div>', unsafe_allow_html=True)

meta_col1, meta_col2, meta_col3 = st.columns(3)
meta_col1.metric("Mode", st.session_state.mode)
algo_label = st.session_state.ai1_algo if st.session_state.mode != "AI vs AI (Auto-play)" else f"X: {st.session_state.ai1_algo} • O: {st.session_state.ai2_algo}"
meta_col2.metric("Algorithm", algo_label)
turn_txt = "Game Over" if st.session_state.winner else f"Turn: {st.session_state.current}"
meta_col3.metric("Status", turn_txt)

# ---------------------- Board UI ----------------------
def cell_label(i: int) -> str:
    v = st.session_state.board[i]
    if st.session_state.win_line and i in st.session_state.win_line:
        if v == "X": return "✨❌✨"
        if v == "O": return "✨⭕✨"
    if v == "X": return "❌"
    if v == "O": return "⭕"
    return " "  # hint: clickable

grid = st.container()
with grid:
    for r in range(3):
        cols = st.columns(3, gap="small")
        for c in range(3):
            i = r*3 + c
            disabled = (st.session_state.board[i] != "") or bool(st.session_state.winner) or (st.session_state.mode == "AI vs AI (Auto-play)")
            if cols[c].button(cell_label(i), key=f"cell_{i}", use_container_width=True, disabled=disabled):
                if st.session_state.mode in ["Human vs Human", "Human vs AI"]:
                    # Only allow when it's a human's turn
                    if st.session_state.mode == "Human vs Human":
                        drop_piece(i, st.session_state.current)
                    else:
                        if st.session_state.current == st.session_state.human_symbol:
                            drop_piece(i, st.session_state.current)

# ---------------------- Human vs AI: trigger AI reply automatically ----------------------
if st.session_state.mode == "Human vs AI" and not st.session_state.winner:
    # If it's AI's turn, take it
    ai_symbol = "O" if st.session_state.human_symbol == "X" else "X"
    if st.session_state.current == ai_symbol:
        ai_move(st.session_state.ai1_algo, ai_symbol)

# ---------------------- AI vs AI Autoplay ----------------------
if st.session_state.mode == "AI vs AI (Auto-play)" and st.session_state.autoplay:
    if not st.session_state.winner:
        current_algo = (
            st.session_state.ai1_algo
            if st.session_state.current == "X"
            else st.session_state.ai2_algo
        )
        ai_move(current_algo, st.session_state.current)

        # ✅ rerun after each move to keep autoplaying
        import time
        time.sleep(st.session_state.ai_delay)  # respect your speed slider
        st.rerun()

# ---------------------- Info & Metrics ----------------------
info = st.container()
with info:
    left, right = st.columns([1,1])
    with left:
        st.markdown('<p style="color: #FFFFFF; font-size: 24px;">Game Info</p>', unsafe_allow_html=True)
        if st.session_state.winner == "Draw":
            st.markdown(f"""<div style="background-color: #68A886;color: white;padding: 0.75rem;border-radius: 8px;border: 1px solid #444;font-size: 1rem;">It's a draw!</div>""",unsafe_allow_html=True)
        elif st.session_state.winner in ["X", "O"]:
            st.markdown(f"""<div style="background-color: #8A4A4A;color: white;padding: 0.75rem;border-radius: 8px;border: 1px solid #444;font-size: 1rem;"><b>{st.session_state.current}</b> lost 😞</div>""",unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background-color: #8A4A4A;color: white;padding: 0.75rem;border-radius: 8px;border: 1px solid #444;font-size: 1rem;"><b>{st.session_state.current}</b>'s turn.</div>""",unsafe_allow_html=True)

        if st.session_state.mode == "Human vs AI":
            st.markdown(f"<p class='white-text'>You are <b>{st.session_state.human_symbol}</b>. "f"AI plays <b>{'O' if st.session_state.human_symbol == 'X' else 'X'}</b> "f"using <b>{st.session_state.ai1_algo}</b>.</p>",unsafe_allow_html=True)
        elif st.session_state.mode == "AI vs AI (Auto-play)":
            st.markdown(f"<p class='white-text'>X uses <b>{st.session_state.ai1_algo}</b>, "f"O uses <b>{st.session_state.ai2_algo}</b>. "f"Use the sidebar to start/stop and adjust speed.</p>",unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: #FFFFFF; font-size: 1rem;">Players take turns making a move.</p>', unsafe_allow_html=True)

    with right:
        st.markdown('<p style="color: #FFFFFF; font-size: 24px;">Preformance (per AI move)</p>', unsafe_allow_html=True)
        if st.session_state.perf_log:
            st.dataframe(st.session_state.perf_log, use_container_width=True)
        else:
            st.markdown('<p style="color: #FFFFFF; font-size: 12px;">AI performance metrics will appear here when the AI makes a move.</p>', unsafe_allow_html=True)

# ---------------------- Footer ----------------------
with st.expander("How scoring works (evaluation function)"):
    st.markdown("""<p class='white-text'>Tic-Tac-Toe is a <b>two-player, zero-sum, deterministic, perfect-information, sequential game</b> played on a 3×3 board. Players take turns marking empty cells with their symbol ('X' or 'O'). The first player to align three symbols in a row (horizontally, vertically, or diagonally) wins. If the board fills up with no winner, the game ends in a draw.</p>""",unsafe_allow_html=True)
