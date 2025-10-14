# Tic-Tac-Toe — Minimax vs Alpha-Beta 

A Streamlit web app that lets you play Tic-Tac-Toe in three modes while comparing Minimax and Alpha-Beta pruning.

# Creation Info

Gabriela del Cristo - CAI4002

## Features
- **Modes**: Human vs Human, Human vs AI, AI vs AI (auto-play).
- **Algorithms**: Plain Minimax and Alpha-Beta pruning.
- **Metrics**: Decision time, nodes explored, nodes pruned, pruning efficiency.
- **UX**: Clear turn indicator, clickable cells, restart & mode/algorithm switching, auto-play with speed control, draw/lose indicator, highlighted current.
- **Extra**: Music that can be toggled on/off.

## Run locally or...

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
## ...visit website

https://crossandcircles.streamlit.app/

## Notes
- Evaluation function: +10 AI win, -10 opponent win, 0 draw; depth is incorporated to prefer quicker wins / slower losses.
- Alpha-Beta pruning metrics include both **nodes explored** and **nodes pruned**.

## Added Files

- README.md
- requirements.txt
- soundgame.mp3
- streamlit_app.py

