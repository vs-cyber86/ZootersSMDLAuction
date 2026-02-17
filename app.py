import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Cricket Performance Dashboard", layout="wide")

# -----------------------------
# League -> tournament mapping
# -----------------------------
LEAGUE_TOURNAMENT = {
    224: ("zooterisdt8", "ISDT 8.0"),
    200: ("zootersmdl2", "SMDL 2.0"),
}

def league_to_tournament(league_val):
    try:
        lg = int(float(str(league_val).strip()))
    except Exception:
        return ("(unknown)", "(Unknown)")
    return LEAGUE_TOURNAMENT.get(lg, (f"league_{lg}", f"League {lg}"))

@st.cache_data
def load_csv_from_repo(filename: str) -> pd.DataFrame:
    p = Path(__file__).resolve().parent / filename
    if not p.exists():
        st.error(f"Missing file in repo: {p.name}. Commit it next to app.py.")
        st.stop()
    return pd.read_csv(p)

def to_float(s):
    return pd.to_numeric(s, errors="coerce")

def overs_to_balls(overs_series: pd.Series) -> pd.Series:
    x = overs_series.astype(str).str.strip()
    parts = x.str.split(".", n=1, expand=True)
    o = pd.to_numeric(parts[0], errors="coerce").fillna(0)
    b = pd.to_numeric(parts[1], errors="coerce").fillna(0)
    return (o * 6 + b).astype(int)

def norm_opp(df: pd.DataFrame) -> pd.DataFrame:
    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].fillna("").astype(str).str.strip()
        df.loc[df["opponent"] == "", "opponent"] = "(Unknown)"
    else:
        df["opponent"] = "(Unknown)"
    return df

def article(word: str) -> str:
    vowels = ["a", "e", "i", "o", "u"]
    word = (word or "").strip()
    if not word:
        return "a"
    return "an" if word[0].lower() in vowels else "a"

def ensure_tournament_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "league" in df.columns:
        keys_labels = df["league"].apply(league_to_tournament)
        df["tournamentkey"] = keys_labels.apply(lambda x: x[0])
        df["tournament"] = keys_labels.apply(lambda x: x[1])
    else:
        df["tournamentkey"] = "(unknown)"
        df["tournament"] = "(Unknown)"
    return df

def infer_opponent_from_match(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
    """
    Best-effort opponent inference:
    - If opponent already exists and has non-empty values, keep it.
    - Else, if matchid exists: for each matchid, if exactly 2 teams appear, map each to the other.
      If more/less than 2 teams, leave Unknown.
    """
    df = df.copy()
    if "opponent" in df.columns and (df["opponent"].fillna("").astype(str).str.strip() != "").any():
        return norm_opp(df)

    df["opponent"] = "(Unknown)"
    if "matchid" not in df.columns or team_col not in df.columns:
        return df

    tmp = df[["matchid", team_col]].copy()
    tmp[team_col] = tmp[team_col].fillna("").astype(str).str.strip()

    teams_by_match = tmp.groupby("matchid")[team_col].apply(lambda s: sorted(set([x for x in s if x])))

    opp_map = {}
    for mid, teams in teams_by_match.items():
        if len(teams) == 2:
            t1, t2 = teams
            opp_map[(mid, t1)] = t2
            opp_map[(mid, t2)] = t1

    df["opponent"] = df.apply(
        lambda r: opp_map.get((r.get("matchid"), str(r.get(team_col) or "").strip()), "(Unknown)"),
        axis=1
    )
    return df

# -----------------------------
# Load data (repo files only)
# -----------------------------
st.title("Cricket Performance Dashboard")

bat = load_csv_from_repo("BattingsZooters.csv")
bowl = load_csv_from_repo("BowlingZooters.csv")

# Add tournament from league (so it never shows blank)
bat = ensure_tournament_cols(bat)
bowl = ensure_tournament_cols(bowl)

# Best-effort opponent inference
bat = infer_opponent_from_match(bat, "battingteam" if "battingteam" in bat.columns else "battingTeam")
bowl = infer_opponent_from_match(bowl, "bowlingteam" if "bowlingteam" in bowl.columns else "bowlingTeam")

# -----------------------------
# Basic cleaning
# -----------------------------
bat["runs"] = to_float(bat.get("runs"))
bat["balls"] = to_float(bat.get("balls"))
bat["fours"] = to_float(bat.get("fours"))
bat["sixes"] = to_float(bat.get("sixes"))
bat["sr"] = to_float(bat.get("sr"))

if "batpos" in bat.columns:
    bat["batpos"] = pd.to_numeric(bat["batpos"], errors="coerce")

if "playername" in bat.columns:
    bat["playername"] = bat["playername"].fillna("").astype(str).str.lower().str.strip()
else:
    bat["playername"] = ""

bowl["runsconceded"] = to_float(bowl.get("runsconceded"))
bowl["wickets"] = to_float(bowl.get("wickets"))
bowl["econ"] = to_float(bowl.get("econ"))
bowl["balls_bowled"] = overs_to_balls(bowl.get("overs", pd.Series(dtype=str)))

if "bowlername" in bowl.columns:
    bowl["bowlername"] = bowl["bowlername"].fillna("").astype(str).str.lower().str.strip()
else:
    bowl["bowlername"] = ""

bat = norm_opp(bat)
bowl = norm_opp(bowl)

# Navigation
tab_player, tab_comparison = st.tabs(["Player Performance", "Player Comparison"])

with tab_player:
    st.subheader("Player performance")

    all_players = sorted(set(bat["playername"]).union(set(bowl["bowlername"])))
    all_players = [p for p in all_players if p]
    player = st.selectbox("Player Name", ["(Select a player)"] + all_players)

    if player == "(Select a player)":
        st.info("Select a player to view performance.")
        st.stop()

    bt = bat[bat["playername"] == player].copy()
    bw = bowl[bowl["bowlername"] == player].copy()

    tourn_options = sorted(set(bt["tournamentkey"]).union(set(bw["tournamentkey"])))
    tourn_labels = {k: bt[bt["tournamentkey"] == k]["tournament"].iloc[0] if (k in bt["tournamentkey"].values) else
                       (bw[bw["tournamentkey"] == k]["tournament"].iloc[0] if (k in bw["tournamentkey"].values) else k)
                    for k in tourn_options}
    tourn_display = [tourn_labels[k] for k in tourn_options]
    display_to_key = {tourn_labels[k]: k for k in tourn_options}

    tourn_choice = st.selectbox("Tournament", ["(All)"] + tourn_display)

    bat_f = bt
    bowl_f = bw
    if tourn_choice != "(All)":
        tkey = display_to_key[tourn_choice]
        bat_f = bat_f[bat_f["tournamentkey"] == tkey]
        bowl_f = bowl_f[bowl_f["tournamentkey"] == tkey]

    st.markdown("### Batting summary")
    if bat_f.empty:
        st.warning("No batting data for this selection.")
    else:
        dismissals = (bat_f.get("howout", "").fillna("").astype(str).str.lower() != "not out").sum()
        inns = len(bat_f)
        runs = bat_f["runs"].sum(skipna=True)
        balls = bat_f["balls"].sum(skipna=True)
        sr = (runs * 100 / balls) if balls else 0
        avg = (runs / dismissals) if dismissals else None

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Inns", int(inns))
        k2.metric("Runs", int(runs) if pd.notna(runs) else 0)
        k3.metric("Avg", f"{avg:.2f}" if avg is not None else "—")
        k4.metric("SR", f"{sr:.2f}" if balls else "—")
        k5.metric("4s/6s", f"{int(bat_f['fours'].sum(skipna=True))}/{int(bat_f['sixes'].sum(skipna=True))}")

        st.markdown("#### Match-wise Batting Summary")
        sort_cols = [c for c in ["tournament", "opponent", "inningsno", "batpos", "matchid"] if c in bat_f.columns]
        bat_detail = bat_f.sort_values(sort_cols) if sort_cols else bat_f
        cols = [c for c in ["tournament", "opponent", "inningsno", "battingteam", "batpos", "runs", "balls", "sr", "fours", "sixes", "howout"] if c in bat_detail.columns]
        st.dataframe(bat_detail[cols], use_container_width=True)

    st.markdown("### Bowling summary")
    if bowl_f.empty:
        st.warning("No bowling data for this selection.")
    else:
        inns_b = len(bowl_f)
        balls_bowled = bowl_f["balls_bowled"].sum()
        overs_equiv = (balls_bowled / 6) if balls_bowled else 0
        runs_c = bowl_f["runsconceded"].sum(skipna=True)
        wkts = bowl_f["wickets"].sum(skipna=True)

        econ = (runs_c / overs_equiv) if overs_equiv else 0
        avg_b = (runs_c / wkts) if wkts else None
        sr_b = (balls_bowled / wkts) if wkts else None

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Spells", int(inns_b))
        k2.metric("Runs", int(runs_c) if pd.notna(runs_c) else 0)
        k3.metric("Wkts", int(wkts) if pd.notna(wkts) else 0)
        k4.metric("Econ", f"{econ:.2f}" if overs_equiv else "—")
        k5.metric("Avg / SR", f"{avg_b:.2f} / {sr_b:.1f}" if (avg_b is not None and sr_b is not None) else "—")

        st.markdown("#### Match-wise Bowling Summary")
        sort_cols = [c for c in ["tournament", "opponent", "inningsno", "matchid"] if c in bowl_f.columns]
        bowl_detail = bowl_f.sort_values(sort_cols) if sort_cols else bowl_f
        cols = [c for c in ["tournament", "opponent", "inningsno", "bowlingteam", "overs", "runsconceded", "wickets", "econ"] if c in bowl_detail.columns]
        st.dataframe(bowl_detail[cols], use_container_width=True)

with tab_comparison:
    st.subheader("Player-wise Overall Comparison")

    all_players_list = sorted(set(bat["playername"]).union(set(bowl["bowlername"])))
    all_players_list = [p for p in all_players_list if p]

    select_all = st.checkbox("Select all players", value=False)
    if select_all:
        selected_players = all_players_list
    else:
        selected_players = st.multiselect(
            "Select players to compare",
            all_players_list,
            default=[all_players_list[0]] if all_players_list else [],
        )

    if not selected_players:
        st.info("Select at least one player to view comparison.")
    else:
        st.markdown("### Batting Comparison")
        bat_comp = bat[bat["playername"].isin(selected_players)].copy()
        if bat_comp.empty:
            st.warning("No batting data for selected players.")
        else:
            bat_summary = bat_comp.groupby("playername", as_index=False).agg(
                inns=("runs", "count"),
                runs=("runs", "sum"),
                balls=("balls", "sum"),
                fours=("fours", "sum"),
                sixes=("sixes", "sum"),
            )
            bat_dismissals = bat_comp.copy()
            bat_dismissals["dismissal"] = (bat_dismissals.get("howout", "").fillna("").astype(str).str.lower() != "not out").astype(int)
            dismissal_counts = bat_dismissals.groupby("playername")["dismissal"].sum()

            bat_summary["avg"] = bat_summary.apply(
                lambda row: (row["runs"] / dismissal_counts.get(row["playername"], 0))
                if dismissal_counts.get(row["playername"], 0) > 0 else None,
                axis=1,
            ).round(2)
            bat_summary["sr"] = (bat_summary["runs"] * 100 / bat_summary["balls"]).round(2)

            display_cols = ["playername", "inns", "runs", "balls", "avg", "sr", "fours", "sixes"]
            st.dataframe(bat_summary.sort_values("runs", ascending=False)[display_cols], use_container_width=True)

        st.markdown("### Bowling Comparison")
        bowl_comp = bowl[bowl["bowlername"].isin(selected_players)].copy()
        if bowl_comp.empty:
            st.warning("No bowling data for selected players.")
        else:
            bowl_summary = bowl_comp.groupby("bowlername", as_index=False).agg(
                spells=("runsconceded", "count"),
                runs=("runsconceded", "sum"),
                balls=("balls_bowled", "sum"),
                wickets=("wickets", "sum"),
            )

            bowl_summary["overs"] = (bowl_summary["balls"] / 6).round(1)
            bowl_summary["econ"] = (bowl_summary["runs"] / (bowl_summary["balls"] / 6)).round(2)
            bowl_summary["avg"] = (bowl_summary["runs"] / bowl_summary["wickets"]).round(2)
            bowl_summary["sr"] = (bowl_summary["balls"] / bowl_summary["wickets"]).round(1)

            display_cols = ["bowlername", "spells", "runs", "overs", "wickets", "avg", "econ", "sr"]
            st.dataframe(bowl_summary.sort_values("wickets", ascending=False)[display_cols], use_container_width=True)
