import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Cricket Performance Dashboard", layout="wide")

# -----------------------------
# Tournament display name mapping
# -----------------------------
TOURNAMENT_LABELS = {
    "zootermt20cb3": "Zooter Pink Ball",
    "zooterisdt8": "Zooter ISDT",
    "jfscdpl202526": "JFSC Dad's",
    "clt20s_pdf": "JFSC CLT20",
}

def tournament_label(k: str) -> str:
    k = str(k)
    return TOURNAMENT_LABELS.get(k, k)

@st.cache_data
def load_csv(uploaded_file, fallback_filename: str):
    """
    If user uploads a CSV, read it.
    Else read fallback file from the same directory as this app.py.
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    p = Path(__file__).resolve().parent / fallback_filename
    if not p.exists():
        st.error(f"Missing file in repo: {p.name}. Commit it or upload it from the sidebar.")
        st.stop()
    return pd.read_csv(p)

def to_float(s):
    return pd.to_numeric(s, errors="coerce")

def overs_to_balls(overs_series: pd.Series) -> pd.Series:
    # cricket overs like 3.3 = 3 overs + 3 balls = 21 balls
    x = overs_series.astype(str).str.strip()
    parts = x.str.split(".", n=1, expand=True)
    o = pd.to_numeric(parts[0], errors="coerce").fillna(0)
    b = pd.to_numeric(parts[1], errors="coerce").fillna(0)
    return (o * 6 + b).astype(int)

def pick_team_mask(series: pd.Series, team_query: str) -> pd.Series:
    q = (team_query or "").strip().lower()
    if not q:
        return pd.Series([True] * len(series), index=series.index)
    return series.astype(str).str.lower().str.contains(q, regex=False)

def norm_opp(df: pd.DataFrame) -> pd.DataFrame:
    # Keep blank opponents as "(Unknown)" for display
    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].fillna("").astype(str).str.strip()
        df.loc[df["opponent"] == "", "opponent"] = "(Unknown)"
    else:
        df["opponent"] = "(Unknown)"
    return df

def article(word: str) -> str:
    """Return the appropriate article (a/an) for a word."""
    vowels = ["a", "e", "i", "o", "u"]
    word = (word or "").strip()
    if not word:
        return "a"
    return "an" if word[0].lower() in vowels else "a"

def generate_player_blurb(player_name: str, bat_stats: dict, bowl_stats: dict, bat_f: pd.DataFrame) -> str:
    """Generate a cricket-specific blurb based on performance metrics and batting positions."""
    runs = bat_stats.get("runs", 0) or 0
    avg = bat_stats.get("avg", 0) or 0
    sr = bat_stats.get("sr", 0) or 0
    bat_inns = bat_stats.get("inns", 0) or 0
    fours = bat_stats.get("fours", 0) or 0
    sixes = bat_stats.get("sixes", 0) or 0

    wickets = bowl_stats.get("wickets", 0) or 0
    econ = bowl_stats.get("econ", 0) or 0
    bowl_inns = bowl_stats.get("spells", 0) or 0

    # ===== BATTING POSITION ANALYSIS =====
    position_insight = ""
    primary_position = ""
    if "batpos" in bat_f.columns and (not bat_f.empty) and (not bat_f["batpos"].isna().all()):
        bat_pos_data = bat_f[bat_f["batpos"].notna()].copy()
        if not bat_pos_data.empty:
            position_counts = bat_pos_data["batpos"].value_counts()
            primary_position_num = int(position_counts.idxmax())

            if primary_position_num <= 2:
                primary_position = "opening batter"
                position_insight = "He is a reliable opener who can set the tone at the start of innings. "
            elif primary_position_num <= 4:
                primary_position = "middle-order batter"
                position_insight = "He is an important middle-order player who provides stability and acceleration. "
            elif primary_position_num <= 6:
                primary_position = "lower middle-order batter"
                position_insight = "He plays in the lower-middle order and can transition from consolidation to aggressive batting. "
            else:
                primary_position = "lower-order batter"
                position_insight = "He contributes valuable support in the lower order with both bat and ball. "

    # ===== BATTING ASSESSMENT =====
    if runs > 1000:
        run_category = "prolific run-scorer with exceptional consistency"
    elif runs > 500:
        run_category = "prolific run-scorer"
    elif runs > 250:
        run_category = "reliable run-maker"
    elif runs > 100:
        run_category = "developing batter with solid fundamentals"
    else:
        run_category = "emerging talent with limited opportunities"

    if sr > 150:
        bat_style_desc = "plays with explosive intent"
    elif sr > 130:
        bat_style_desc = "plays with aggressive intent"
    elif sr > 110:
        bat_style_desc = "balances aggression with caution"
    elif sr > 90:
        bat_style_desc = "focuses on steady accumulation"
    else:
        bat_style_desc = "plays a cautious game"

    if avg > 40:
        avg_quality = "exceptional average"
    elif avg > 30:
        avg_quality = "strong average"
    elif avg > 20:
        avg_quality = "solid average"
    elif avg > 10:
        avg_quality = "decent average"
    else:
        avg_quality = "developing consistency"

    total_boundaries = fours + sixes
    boundary_ratio = (total_boundaries / runs) if runs > 0 else 0

    if sixes > fours * 0.5 and sixes > 0:
        aggression = "demonstrates strong six-hitting ability"
    elif boundary_ratio > 0.25:
        aggression = "is a strong boundary hitter"
    elif boundary_ratio > 0.15:
        aggression = "rotates the strike well with regular boundaries"
    else:
        aggression = "relies on singles and twos"

    # ===== AREAS OF IMPROVEMENT =====
    improvements = []
    if runs < 100 and bat_inns > 0:
        improvements.append("needs to increase run accumulation")
    elif avg < 20 and bat_inns > 3:
        improvements.append("should focus on consistency and reducing dismissals")
    elif avg < 30 and bat_inns > 3:
        improvements.append("needs to elevate his average")

    if sr < 100 and bat_inns > 5:
        improvements.append("should work on increasing strike rate")
    elif sr > 160 and bat_inns > 5:
        improvements.append("needs to balance aggression with stability")

    if boundary_ratio < 0.10 and runs > 50:
        improvements.append("should look to score more boundaries")
    elif sixes == 0 and sr > 120:
        improvements.append("should work on developing six-hitting ability")

    if bat_inns > 5 and (not bat_f.empty) and ("howout" in bat_f.columns):
        not_outs = (bat_f["howout"].fillna("").str.lower() == "not out").sum()
        dismissal_rate = 1 - (not_outs / bat_inns) if bat_inns else 0
        if dismissal_rate > 0.8:
            improvements.append("needs to work on converting starts into substantial scores")

    bowl_text = ""
    bowl_improvements = []
    if bowl_inns > 0 and wickets > 0:
        if wickets > 50:
            bowl_category = "a leading bowler"
        elif wickets > 20:
            bowl_category = "an impactful bowler"
        elif wickets > 10:
            bowl_category = "a useful bowler"
        else:
            bowl_category = "an occasional bowler"

        if econ < 5:
            econ_quality = "excellent economy rate"
        elif econ < 6.5:
            econ_quality = "good economy rate"
        elif econ < 8:
            econ_quality = "acceptable economy rate"
        else:
            econ_quality = "needs to work on economy"

        bowl_text = (
            f" As {bowl_category}, he bowls with {econ_quality} (Econ: {econ:.2f}) "
            f"and has taken {int(wickets)} wickets across {int(bowl_inns)} spells."
        )

        if econ > 8 and bowl_inns > 3:
            bowl_improvements.append("should work on tightening bowling lines and lengths")

        if wickets < 5 and bowl_inns > 10:
            bowl_improvements.append("needs to improve wicket-taking ability")
        elif wickets < 10 and bowl_inns > 15:
            bowl_improvements.append("should focus on taking more wickets")

    # ===== FINAL BLURB =====
    blurb = f"**{player_name}** is {article(run_category)} {run_category}"
    if primary_position:
        blurb += f" and {primary_position}"
    blurb += f". {position_insight}"
    blurb += (
        f"With {article(avg_quality)} {avg_quality} of {avg:.2f} and a strike rate of {sr:.0f}, "
        f"{player_name} {bat_style_desc} and {aggression}."
    )

    if bowl_text:
        blurb += bowl_text
    else:
        if bat_inns > 0:
            blurb += f" He has played {int(bat_inns)} innings in this period with a primary focus on batting."

    all_improvements = improvements + bowl_improvements
    if all_improvements:
        blurb += "\n\n**Areas for Improvement:** " + ", ".join(all_improvements) + "."

    return blurb

st.title("Cricket Performance Dashboard")

# Sidebar upload (optional)
with st.sidebar:
    st.header("Data")
    up_bat = st.file_uploader("Upload BattingsZooters.csv", type=["csv"], key="bat")
    up_bowl = st.file_uploader("Upload BowlingZooters.csv", type=["csv"], key="bowl")
    st.caption("If you don’t upload, the app will try reading the repo files in the same folder as app.py.")
    st.divider()

# IMPORTANT: use your current repo filenames here
bat = load_csv(up_bat, "BattingsZooters.csv")
bowl = load_csv(up_bowl, "BowlingZooters.csv")

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

# Normalize player names to lowercase for consistent deduplication
if "playername" in bat.columns:
    bat["playername"] = bat["playername"].fillna("").astype(str).str.lower().str.strip()

bowl["runsconceded"] = to_float(bowl.get("runsconceded"))
bowl["wickets"] = to_float(bowl.get("wickets"))
bowl["econ"] = to_float(bowl.get("econ"))
bowl["balls_bowled"] = overs_to_balls(bowl.get("overs", pd.Series(dtype=str)))

# Normalize bowler names to lowercase for consistent deduplication
if "bowlername" in bowl.columns:
    bowl["bowlername"] = bowl["bowlername"].fillna("").astype(str).str.lower().str.strip()

bat = norm_opp(bat)
bowl = norm_opp(bowl)

# Add friendly tournament label
if "tournamentkey" in bat.columns:
    bat["tournament"] = bat["tournamentkey"].map(lambda x: tournament_label(x))
else:
    bat["tournament"] = "(Unknown)"

if "tournamentkey" in bowl.columns:
    bowl["tournament"] = bowl["tournamentkey"].map(lambda x: tournament_label(x))
else:
    bowl["tournament"] = "(Unknown)"

# Navigation
tab_player, tab_comparison = st.tabs(["Player Performance", "Player Comparison"])

# -------------------------
# Tab 1: Player performance
# -------------------------
with tab_player:
    st.subheader("Player performance")

    all_players = sorted(set(bat.get("playername", pd.Series(dtype=str))).union(set(bowl.get("bowlername", pd.Series(dtype=str)))))
    player = st.selectbox("Player Name", ["(Select a player)"] + [p for p in all_players if p])

    if player == "(Select a player)":
        st.info("Select a player to view performance.")
        st.stop()

    # Tournament options depend on selected player
    bt = bat[bat["playername"] == player].copy()
    bw = bowl[bowl["bowlername"] == player].copy()
    tourn_options = sorted(set(bt.get("tournamentkey", pd.Series(dtype=str))).union(set(bw.get("tournamentkey", pd.Series(dtype=str)))))
    tourn_display = [tournament_label(t) for t in tourn_options if str(t) != "nan"]
    display_to_key = {tournament_label(t): t for t in tourn_options if str(t) != "nan"}

    tourn_choice = st.selectbox("Tournament", ["(All)"] + tourn_display)

    bat_f = bt
    bowl_f = bw

    if tourn_choice != "(All)":
        tkey = display_to_key[tourn_choice]
        bat_f = bat_f[bat_f["tournamentkey"] == tkey]
        bowl_f = bowl_f[bowl_f["tournamentkey"] == tkey]

    # Blurb inputs
    bat_stats = {}
    bowl_stats = {}

    if not bat_f.empty:
        dismissals = (bat_f["howout"].fillna("").str.lower() != "not out").sum() if "howout" in bat_f.columns else 0
        runs_sum = bat_f["runs"].sum(skipna=True)
        balls_sum = bat_f["balls"].sum(skipna=True)
        bat_stats = {
            "inns": len(bat_f),
            "runs": runs_sum,
            "avg": (runs_sum / dismissals) if dismissals else None,
            "sr": (runs_sum * 100 / balls_sum) if balls_sum else 0,
            "fours": bat_f["fours"].sum(skipna=True),
            "sixes": bat_f["sixes"].sum(skipna=True),
        }

    if not bowl_f.empty:
        balls_bowled = bowl_f["balls_bowled"].sum()
        bowl_stats = {
            "spells": len(bowl_f),
            "wickets": bowl_f["wickets"].sum(skipna=True),
            "econ": (bowl_f["runsconceded"].sum(skipna=True) / (balls_bowled / 6)) if balls_bowled else 0,
        }

    if bat_stats or bowl_stats:
        st.markdown(generate_player_blurb(player, bat_stats, bowl_stats, bat_f))
        st.markdown("---")

    # Batting summary
    st.markdown("### Batting summary")
    if bat_f.empty:
        st.warning("No batting data for this selection.")
    else:
        dismissals = (bat_f["howout"].fillna("").str.lower() != "not out").sum() if "howout" in bat_f.columns else 0
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
        sort_cols = [c for c in ["tournamentkey", "opponent", "inningsno", "batpos", "matchid"] if c in bat_f.columns]
        bat_detail = bat_f.sort_values(sort_cols) if sort_cols else bat_f

        cols = [c for c in ["tournament", "opponent", "inningsno", "battingteam", "batpos", "runs", "balls", "sr", "fours", "sixes", "howout"] if c in bat_detail.columns]
        st.dataframe(bat_detail[cols], use_container_width=True)

    # Bowling summary
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
        sort_cols = [c for c in ["tournamentkey", "opponent", "inningsno", "matchid"] if c in bowl_f.columns]
        bowl_detail = bowl_f.sort_values(sort_cols) if sort_cols else bowl_f

        cols = [c for c in ["tournament", "opponent", "inningsno", "bowlingteam", "overs", "runsconceded", "wickets", "econ"] if c in bowl_detail.columns]
        st.dataframe(bowl_detail[cols], use_container_width=True)

# -------------------------
# Tab 2: Player-wise overall comparison
# -------------------------
with tab_comparison:
    st.subheader("Player-wise Overall Comparison")

    all_players_list = sorted(set(bat.get("playername", pd.Series(dtype=str))).union(set(bowl.get("bowlername", pd.Series(dtype=str)))))
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
