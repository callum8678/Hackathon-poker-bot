"""
PleaseFold v2 -- "Please fold. We'd rather not show you the river."

A 6-max No-Limit Hold'em bot for the NUCATS table. Everything lives in this
one file and uses only the standard library plus the provided `pokerbot`
helpers, so it needs nothing installed.

How it plays
    * Preflop: position-aware open / 3-bet / call / fold ranges built on a
      169-hand strength ranking, push-or-fold when the stack gets short, and
      an equity check (against the raiser's likely range) when facing a big
      raise or all-in.
    * Postflop: a Monte Carlo equity estimate against the opponents' likely
      ranges (narrowed by what they did this hand), then pot-odds / EV
      decisions: value-bet, call, raise or fold, with pot-fraction sizing.
    * v2: in the closing hands it compares its stack with the biggest rival's
      and plays tighter in big pots when leading (looser when far behind).
    * Optional extras (off by default, see the flags below): trapping with
      aces/kings (smaller raises, flat-calls, slow-play), selective bluffs,
      a light per-seat opponent model.
    * Safe by construction: a time budget on the simulation and a fallback
      move if anything goes wrong.

It only uses what the table shows every bot: its own cards, the board, the
pot, stacks, legal actions and the public action history.

Run it (the organiser will tell you the host and port):
    python3 PleaseFold.py --host 192.168.1.20 --port 8000
"""
import random
import time

from pokerbot import all_in, call, check, fold, raise_to, run

BOT_NAME = "PleaseFoldV2"

# ----------------------------------------------------------------------
# Tunables (kept as module constants so they are easy to adjust)
EQ_POST = True      # Monte Carlo equity + pot-odds play after the flop
SIZING = True       # pot-fraction bet sizing (else flat half pot)
# The three options below are built but switched off: in testing none of them
# showed a measurable gain (confidence intervals included zero), so the
# validated, simpler configuration is the default.
BLUFF = False       # capped bluffs and semi-bluffs
MODEL = False       # per-seat opponent model
ENDGAME = False     # chip-position adjustment in the last few hands
LEAD = True         # protect a chip lead / chase when behind in the closing hands
LEAD_HANDS = 6      # how many hands from the end the lead logic starts to ramp up
TRAP_PRE = 0        # 0 off; 1 smaller raises with aces/kings; 2 also flat-call / limp sometimes
TRAP_POST = False   # sometimes slow-play a near-certain winner to let others bet
TRAP_P = 0.5        # chance of taking the trap line when it applies
MC_N = 1500         # max Monte Carlo samples per decision
TIME_SOFT = 0.8     # seconds: stop sampling after this (the table allows 5)
PUSH_BB = 12        # push-or-fold when the effective stack is this many bb or less

_RNG = random.Random()

# ----------------------------------------------------------------------
# Cards and hand evaluation (own implementation, validated against pokerkit)
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"


def _parse(card):
    return _RANKS.index(card[0]) * 4 + _SUITS.index(card[1])


def _build_straights():
    table = [-1] * 8192
    for m in range(8192):
        best = -1
        for hi in range(12, 3, -1):
            if all(m >> (hi - i) & 1 for i in range(5)):
                best = hi
                break
        if best < 0 and (m & 0x1000) and (m & 0xF) == 0xF:   # A-2-3-4-5
            best = 3
        table[m] = best
    return table


_STRAIGHT = _build_straights()


def _top5(mask):
    out, n, r = 0, 0, 12
    while r >= 0 and n < 5:
        if mask >> r & 1:
            out = (out << 4) | r
            n += 1
        r -= 1
    while n < 5:
        out <<= 4
        n += 1
    return out


def _pack(ranks):
    out = 0
    for i in range(5):
        out = (out << 4) | (ranks[i] if i < len(ranks) else 0)
    return out


def _evaluate(cards):
    """Score for 3-7 card ints (rank*4+suit). Bigger is better; score >> 20 is
    the hand category (0 high card ... 8 straight flush)."""
    rc = [0] * 13
    sm = [0, 0, 0, 0]
    sc = [0, 0, 0, 0]
    mask = 0
    for c in cards:
        r = c >> 2
        s = c & 3
        rc[r] += 1
        sm[s] |= 1 << r
        sc[s] += 1
        mask |= 1 << r
    flush = -1
    for s in range(4):
        if sc[s] >= 5:
            flush = s
            break
    if flush >= 0:
        sf = _STRAIGHT[sm[flush]]
        if sf >= 0:
            return (8 << 20) | (sf << 16)
    quads, trips, pairs, singles = [], [], [], []
    for r in range(12, -1, -1):
        n = rc[r]
        if n == 1:
            singles.append(r)
        elif n == 2:
            pairs.append(r)
        elif n == 3:
            trips.append(r)
        elif n == 4:
            quads.append(r)
    if quads:
        rest = trips + pairs + singles
        return (7 << 20) | (quads[0] << 16) | ((max(rest) if rest else 0) << 12)
    if trips and (len(trips) > 1 or pairs):
        return (6 << 20) | (trips[0] << 16) | (max(trips[1:] + pairs) << 12)
    if flush >= 0:
        return (5 << 20) | _top5(sm[flush])
    st = _STRAIGHT[mask]
    if st >= 0:
        return (4 << 20) | (st << 16)
    if trips:
        return (3 << 20) | _pack([trips[0]] + singles[:2])
    if len(pairs) >= 2:
        rest = pairs[2:] + singles
        return (2 << 20) | (pairs[0] << 16) | (pairs[1] << 12) | ((max(rest) if rest else 0) << 8)
    if pairs:
        return (1 << 20) | _pack([pairs[0]] + singles[:3])
    return _pack(singles[:5])


# ----------------------------------------------------------------------
# Preflop hand ranking: the 169 starting-hand classes ordered by equity
# against three random hands (generated offline by simulation).
_PRE_ORDER = (
    "AA KK QQ JJ TT AKs 99 AQs AJs KQs AKo 88 ATs AQo KJs KQo AJo KTs QJs A9s "
    "77 KJo A8s QTs JTs ATo K9s A7s KTo A5s QJo Q9s 66 A4s A6s K8s QTo A9o J9s "
    "JTo T9s K7s A3s A8o A2s K9o J8s Q8s T8s K6s A7o Q9o 55 K5s J9o 98s T9o K4s "
    "Q7s A5o A6o K8o A4o A3o Q6s K3s 87s K7o T7s J7s 97s 44 Q5s Q4s Q8o K2s A2o "
    "J8o 76s K6o 86s J6s 98o Q2s J5s T8o 96s Q3s K5o T6s Q7o 33 J4s 75s T5s J3s "
    "65s 85s Q6o K4o J7o 87o K3o 95s 97o T7o 54s J2s Q5o T4s K2o Q4o 74s 84s 64s "
    "T3s 76o 53s 22 T6o J6o 94s T2s 86o Q3o 93s J5o 43s 63s 96o 73s 65o J4o Q2o "
    "75o 83s 92s 85o T5o 52s 82s 95o J3o 62s T4o 54o J2o 42s 72s 64o 32s 84o T3o "
    "T2o 74o 94o 93o 53o 43o 63o 73o 92o 83o 52o 42o 62o 82o 72o 32o"
).split()


def _build_preflop_tables():
    top = {}
    cum = 0
    for nm in _PRE_ORDER:
        hi, lo = _RANKS.index(nm[0]), _RANKS.index(nm[1])
        suited = nm.endswith("s")
        k = 6 if hi == lo else (4 if suited else 12)
        top[(hi, lo, suited)] = (cum + k / 2.0) / 1326.0
        cum += k
    table = [1.0] * 2704
    for c1 in range(52):
        for c2 in range(52):
            if c1 != c2:
                r1, r2 = c1 >> 2, c2 >> 2
                hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
                suited = (c1 & 3) == (c2 & 3) and hi != lo
                table[c1 * 52 + c2] = top[(hi, lo, suited)]
    return table


_TOP = _build_preflop_tables()   # _TOP[c1*52+c2] = fraction of hands better (0 = AA)

# open-raise range (fraction of hands) by number of players still to act
_OPEN = {5: 0.16, 4: 0.20, 3: 0.27, 2: 0.42, 1: 0.45}
_PUSH = {1: 0.50, 2: 0.38, 3: 0.30, 4: 0.25, 5: 0.22}
# vs. our estimate of how wide each kind of preflop action is (fraction of hands)
_RAISE_CUT = {1: 0.30, 2: 0.09, 3: 0.04, 4: 0.025}
_CALL_CUT = {0: 0.70, 1: 0.32, 2: 0.11, 3: 0.05}
# postflop: weight (acceptance prob.) of an opponent holding class k by action;
# class 0 = nothing (or only a kicker), 1 = pair, 2+ = two pair or better
_BET_W = {"flop": (0.35, 0.80), "turn": (0.25, 0.65), "river": (0.15, 0.50)}
_CALL_W = {"flop": (0.50, 1.0), "turn": (0.40, 1.0), "river": (0.30, 0.90)}
# value-bet / raise equity needed, by number of opponents still in the hand
_VALUE = (0.60, 0.46, 0.38, 0.33, 0.30)
_RAISE = (0.74, 0.62, 0.55, 0.50, 0.46)

# ----------------------------------------------------------------------
# Per-match memory (keyed so several bots in one process don't collide)
_MEM = {}


def _memory(state):
    raw = state.raw or {}
    key = (raw.get("match_id"), state.my_seat)
    m = _MEM.get(key)
    if m is None:
        if len(_MEM) > 64:
            _MEM.clear()
        m = {"hand": -1, "seen": 0, "bb": 0, "opp": {}}
        _MEM[key] = m
    if m["hand"] != state.hand_number:
        m["hand"], m["seen"] = state.hand_number, 0
        for p in state.opponents:
            if p.status != "out":
                _opp(m, p.seat)["hands"] += 1
    for mv in state.history:
        if mv.action == "big_blind" and mv.amount > m["bb"]:
            m["bb"] = mv.amount
    return m


def _opp(m, seat):
    o = m["opp"].get(seat)
    if o is None:
        o = {"hands": 0, "vpip": 0, "pfr": 0, "bets": 0, "calls": 0, "vpip_hand": -1}
        m["opp"][seat] = o
    return o


def _update_model(m, state):
    """Fold the not-yet-counted history entries into the per-seat counters."""
    hist = state.history
    for i, mv in enumerate(hist):
        if i < m["seen"] or mv.seat == state.my_seat:
            continue
        o = _opp(m, mv.seat)
        if mv.street == "preflop":
            if mv.action in ("call", "raise") and o["vpip_hand"] != state.hand_number:
                o["vpip"] += 1
                o["vpip_hand"] = state.hand_number
            if mv.action == "raise":
                o["pfr"] += 1
        else:
            if mv.action == "raise":
                o["bets"] += 1
            elif mv.action == "call":
                o["calls"] += 1
    m["seen"] = len(hist)


# ----------------------------------------------------------------------
# Table geometry and history parsing
class _Ctx:
    pass


def _build_ctx(state, m):
    c = _Ctx()
    c.s = state
    c.me = state.my_seat
    c.mine = [_parse(x) for x in state.my_cards]
    c.board = [_parse(x) for x in state.board]
    c.bb = m["bb"] or 20
    c.pot = state.pot
    c.to_call = state.to_call
    c.stack = state.my_stack
    c.total = state.my_stack + state.my_bet           # chips I had at the start of this street
    c.opps = [p for p in state.opponents if p.in_hand]
    c.n = len(c.opps)
    seats = sorted([p.seat for p in state.opponents if p.status != "out"] + [state.my_seat])
    c.nlive = len(seats)
    bi = seats.index(state.button) if state.button in seats else len(seats) - 1
    rot = seats[bi + 1:] + seats[:bi + 1]          # SB first ... button last
    c.rot = rot
    pf = [rot[1], rot[0]] if c.nlive == 2 else rot[2:] + rot[:2]
    c.pf_order = pf
    status = {p.seat: p.status for p in state.opponents}
    after = pf[pf.index(state.my_seat) + 1:] if state.my_seat in pf else []
    c.behind = sum(1 for s in after if status.get(s) in ("active", "all_in"))
    if c.nlive == 2:
        c.is_bb = state.my_seat == rot[0]
        c.is_sb = state.my_seat == rot[1]
    else:
        c.is_bb = state.my_seat == rot[1]
        c.is_sb = state.my_seat == rot[0]
    c.is_btn = state.my_seat == state.button
    c.ip = all(rot.index(o.seat) < rot.index(state.my_seat) for o in c.opps) if c.opps else True
    hi_stack = max([o.stack + o.bet for o in c.opps], default=0)
    c.eff = min(c.total, hi_stack) if c.opps else c.total
    c.hands_left = state.hands_in_match - state.hand_number + 1
    _parse_history(c, state)
    return c


def _parse_history(c, state):
    """Preflop raise level, limpers, per-seat range cutoffs and postflop action weights."""
    level = 0
    limpers = 0
    callers = 0
    last_raise_to = 0
    last_raiser = None
    my_raises = 0
    cut = {}
    street_acts = {}                 # (street, seat) -> {"raise": bool, "call": bool}
    for mv in state.history:
        if mv.street == "preflop":
            a = mv.action
            if a == "raise":
                level += 1
                last_raise_to = mv.amount
                last_raiser = mv.seat
                callers = 0
                if mv.seat == c.me:
                    my_raises += 1
                else:
                    cut[mv.seat] = min(cut.get(mv.seat, 1.0), _RAISE_CUT.get(level, 0.025))
            elif a == "call":
                if level == 0:
                    limpers += 1
                else:
                    callers += 1
                if mv.seat != c.me:
                    cut[mv.seat] = min(cut.get(mv.seat, 1.0), _CALL_CUT.get(min(level, 3), 0.05))
        elif mv.action in ("raise", "call", "check"):
            d = street_acts.setdefault((mv.street, mv.seat), {"raise": False, "call": False})
            if mv.action == "raise":
                d["raise"] = True
            elif mv.action == "call":
                d["call"] = True
    c.level = level
    c.limpers = limpers
    c.callers = callers
    c.last_raise_to = last_raise_to
    c.last_raiser = last_raiser
    c.my_raises = my_raises
    c.pf_aggr_me = last_raiser == c.me
    c.cut = cut
    c.street_acts = street_acts


def _opp_ranges(c, m):
    """[(cutoff, class-weights or None)] for each opponent still in the hand."""
    out = []
    streets = ("flop", "turn", "river")
    for o in c.opps:
        cut = c.cut.get(o.seat, 1.0)
        if MODEL:
            cut = min(1.0, cut * _loose_factor(m, o.seat))
        w = None
        if c.board:
            w = [1.0] * 9
            for st in streets:
                d = c.street_acts.get((st, o.seat))
                if not d:
                    continue
                if d["raise"]:
                    f = _BET_W[st]
                elif d["call"]:
                    f = _CALL_W[st]
                else:
                    continue
                w[0] *= f[0]
                w[1] *= f[1]
            if MODEL:
                agg = _aggression(m, o.seat)
                if agg > 0.5:                      # an aggressive seat bets wider
                    w[0] = min(1.0, w[0] * 1.6)
                    w[1] = min(1.0, w[1] * 1.2)
        out.append((cut, w))
    return out


def _loose_factor(m, seat):
    o = m["opp"].get(seat)
    if not o or o["hands"] < 6:
        return 1.0
    vpip = o["vpip"] / o["hands"]
    return max(0.6, min(1.5, 0.5 + vpip * 1.6)) if vpip > 0 else 0.8


def _aggression(m, seat):
    o = m["opp"].get(seat)
    if not o or o["bets"] + o["calls"] < 5:
        return 0.3
    return o["bets"] / (o["bets"] + o["calls"])


def _is_station(m, seat):
    o = m["opp"].get(seat)
    if not o or o["hands"] < 6:
        return False
    return o["vpip"] / o["hands"] > 0.55 and _aggression(m, seat) < 0.3 and o["calls"] >= 4


# ----------------------------------------------------------------------
# Monte Carlo equity against opponent ranges
def _equity(mine, board, ranges, t_end, n_max):
    used = set(mine) | set(board)
    rem = [x for x in range(52) if x not in used]
    need = 5 - len(board)
    bcat = (_evaluate(board) >> 20) if len(board) >= 3 else 0
    rnd = _RNG.random
    samp = _RNG.sample
    perf = time.perf_counter
    won = 0.0
    n = 0
    top = _TOP
    evaluate = _evaluate
    nrem = len(rem)
    while n < n_max:
        if (n & 63) == 63 and perf() > t_end:
            break
        taken = []
        hands = []
        for cut, w in ranges:
            a = b = -1
            for _ in range(12):
                a, b = samp(rem, 2)
                if a in taken or b in taken:
                    continue
                if top[a * 52 + b] > cut and rnd() > 0.03:
                    continue
                if w is not None:
                    cl = evaluate([a, b] + board) >> 20
                    if cl <= bcat:
                        cl = 0
                    if rnd() > w[cl]:
                        continue
                break
            else:
                while a in taken or b in taken or a == b:
                    a, b = samp(rem, 2)
            taken.append(a)
            taken.append(b)
            hands.append((a, b))
        full = list(board)
        while len(full) < 5:
            x = rem[int(rnd() * nrem)]
            if x not in taken and x not in full:
                full.append(x)
        mine_score = evaluate(mine + full)
        ties = 1
        lost = False
        for a, b in hands:
            sc = evaluate([a, b] + full)
            if sc > mine_score:
                lost = True
                break
            if sc == mine_score:
                ties += 1
        if not lost:
            won += 1.0 / ties
        n += 1
    return won / n if n else 0.5


# ----------------------------------------------------------------------
# Move helpers
def _legal_raise(state, amount):
    """Clamp a raise-to amount into the legal range."""
    return raise_to(int(max(state.min_raise, min(state.max_raise, round(amount)))))


def _passive(state):
    return check() if state.can_check else fold()


def _jam_or_call(state):
    if state.can_raise:
        return all_in()
    return call() if state.to_call else check()


# ----------------------------------------------------------------------
# Preflop
def _preflop(c, m, t_end):
    s = c.s
    top = _TOP[c.mine[0] * 52 + c.mine[1]]
    bb = c.bb
    eff_bb = c.eff / float(bb)
    to_call = c.to_call
    shift = _shift(c, m)
    premium = top <= 0.0075                 # aces and kings

    # short stack: push or fold
    if eff_bb <= PUSH_BB and c.level == 0:
        thr = _PUSH.get(min(5, max(1, c.behind)), 0.22) * (1.15 if eff_bb <= 8 else 1.0)
        if c.is_bb and c.limpers == 0 and s.can_check:
            return check()
        if top <= thr - shift * 0.5:
            return _jam_or_call(s)
        return _passive(s)

    if c.level == 0:
        # nobody has raised yet
        if c.is_bb:
            if s.can_check:
                if TRAP_PRE >= 2 and premium and c.limpers and _RNG.random() < TRAP_P:
                    return check()
                if c.limpers and top <= 0.16 and s.can_raise:
                    return _legal_raise(s, (4 + c.limpers) * bb)
                return check()
        thr = _OPEN.get(min(5, max(1, c.behind)), 0.2)
        if c.is_sb and c.nlive > 2:
            thr = 0.40 if not c.limpers else 0.22
        if c.nlive == 2 and c.is_sb:
            thr = 0.62
        if c.limpers:
            thr = min(thr, 0.30) + 0.02
        thr += shift
        if TRAP_PRE >= 2 and premium and not c.limpers and not s.can_check                 and _RNG.random() < TRAP_P:
            return call()                   # limp with a monster, hoping someone raises
        if top <= thr and s.can_raise:
            small = TRAP_PRE >= 1 and premium
            return _legal_raise(s, ((2.2 if small else 2.5) + c.limpers) * bb)
        if s.can_check:
            return check()
        # limp / complete cheaply with playable hands
        if to_call <= bb and top <= (0.50 if c.is_sb else 0.28):
            return call()
        return fold()

    # facing a raise (level >= 1)
    if to_call <= 0:
        return check() if s.can_check else _passive(s)
    need = to_call / float(c.pot + to_call) if to_call else 0.0
    commit = to_call / float(max(1, c.total))
    big = commit >= 0.30 or c.level >= 3 or eff_bb <= 15 or to_call >= c.stack
    if big:
        ranges = [(c.cut.get(o.seat, 0.5), None) for o in c.opps]
        if MODEL:
            ranges = [(min(1.0, cu * _loose_factor(m, o.seat)), None) for (cu, _), o in zip(ranges, c.opps)]
        if eff_bb <= 15:
            ranges = [(max(cu, 0.30), None) for cu, _ in ranges]
        eq = _equity(c.mine, c.board, ranges, t_end, 700)
        if eq >= need + 0.02 + shift:
            if eq >= 0.60 and s.can_raise and c.level <= 3 and to_call < c.stack:
                return all_in()
            return call()
        return fold()

    v3 = 0.040 if c.level == 1 else 0.026
    if c.last_raise_to >= 5 * bb:
        v3 *= 0.7
    if c.callers:
        v3 += 0.01
    v3 -= shift * 0.5
    if top <= v3 and s.can_raise:
        if TRAP_PRE >= 2 and premium and c.level == 1 and not c.callers and _RNG.random() < TRAP_P:
            return call()                   # flat-call and let them keep betting
        size = c.last_raise_to * (3.0 if c.ip or c.is_btn else 3.6) + c.callers * bb
        if TRAP_PRE >= 1 and premium and c.level == 1:
            size = c.last_raise_to * 2.4 + c.callers * bb
        if c.level >= 2:
            size = c.last_raise_to * 2.4
        if size >= 0.45 * c.total:
            return all_in()
        return _legal_raise(s, size)
    if c.level == 1:
        if c.is_bb:
            cthr = 0.34 - 0.04 * max(0.0, c.last_raise_to / float(bb) - 2.5)
            cthr = max(0.12, min(0.34, cthr))
        elif c.is_sb:
            cthr = 0.10
        elif c.ip or c.is_btn:
            cthr = 0.17
        else:
            cthr = 0.12
        cthr += 0.02 * min(3, c.callers)
        cthr *= max(0.35, 1.0 - 1.5 * commit)
    else:
        cthr = 0.06
    cthr -= shift
    if top <= cthr:
        return call()
    return fold()


# ----------------------------------------------------------------------
# Chip-position adjustment (positive = play tighter)
def _shift(c, m):
    total = 0.0
    if ENDGAME and c.hands_left <= 3:
        mine = c.s.my_stack + c.s.my_bet
        others = [o.stack + o.bet for o in c.s.opponents if o.status != "out"]
        if others:
            lead = max(others)
            if mine >= 1.5 * lead:
                total += 0.04 * (1.0 + (3 - c.hands_left) * 0.3)
            elif mine < 0.6 * lead and c.hands_left <= 2:
                total -= 0.03
    if LEAD:
        # the more of my stack a decision puts at risk, the more the score matters
        stake = min(1.0, 0.3 + 2.5 * c.to_call / float(max(1, c.total)))
        total += _lead_risk(c) * stake
    return total


def _lead_risk(c):
    """Positive = protect a lead (play tighter); negative = need to chase (loosen up).
    Ramps up over the last LEAD_HANDS hands, comparing my stack with the biggest rival's."""
    hl = c.hands_left
    if hl > LEAD_HANDS:
        return 0.0
    mine = c.s.my_stack + c.s.my_bet
    others = [o.stack + o.bet for o in c.s.opponents if o.status != "out"]
    if not others:
        return 0.0
    top = max(others)
    ratio = mine / float(top) if top > 0 else 9.0
    ramp = (LEAD_HANDS - hl + 1) / float(LEAD_HANDS)
    if ratio >= 1.0:
        edge = min(1.0, (ratio - 1.0) / 0.5 + 0.2)
        return 0.07 * ramp * edge
    if hl <= 3 and ratio < 0.7:
        return -0.05 * ramp
    return 0.0


# ----------------------------------------------------------------------
# Postflop
def _postflop(c, m, t_end):
    s = c.s
    n = max(1, c.n)
    ranges = _opp_ranges(c, m)
    shift = _shift(c, m)
    street = s.street
    pot = c.pot
    to_call = c.to_call
    stack = c.stack

    eq = _equity(c.mine, c.board, ranges, t_end, MC_N)
    mine_cat = _evaluate(c.mine + c.board) >> 20
    board_cat = _evaluate(c.board) >> 20
    made = mine_cat > board_cat
    nn = min(n, 5) - 1
    value_thr = _VALUE[nn] + shift + (0.0 if c.ip else 0.02)
    raise_thr = _RAISE[nn] + shift + (0.0 if c.ip else 0.02)

    if to_call > 0:
        need = to_call / float(pot + to_call)
        all_in_call = to_call >= stack
        left_after = stack - to_call
        margin = 0.01 if all_in_call or left_after <= 0.25 * (pot + to_call) else 0.03
        margin += 0.015 * (n - 1) + (0.015 if street == "river" else 0.0) - (0.01 if c.ip else 0.0) + shift
        if TRAP_POST and eq >= 0.90 and street != "river" and n <= 2 and not all_in_call                 and _RNG.random() < TRAP_P:
            return call()                   # slow-play: raise on a later street
        if s.can_raise and eq >= raise_thr and not all_in_call:
            frac = 0.75 if SIZING else 1.0
            target = c.s.my_bet + to_call + (pot + to_call) * frac
            if not SIZING:
                target = c.s.my_bet + to_call * 3
            if target >= 0.55 * c.total or (stack - (target - c.s.my_bet)) <= 0.35 * (pot + target):
                return all_in()
            return _legal_raise(s, target)
        if eq >= need + margin:
            return call()
        return fold()

    # no bet to face: bet or check
    if not s.can_raise:
        return check()
    bet = None
    if TRAP_POST and eq >= 0.90 and street == "flop" and n == 1 and _RNG.random() < TRAP_P:
        return check()                      # slow-play: let them bet or catch up
    if eq >= value_thr:
        if SIZING:
            if eq >= 0.85:
                frac = 0.75 if street != "river" else 0.9
            elif eq >= 0.72:
                frac = 0.66
            elif eq >= 0.62:
                frac = 0.5
            else:
                frac = 0.36
            if n >= 2:
                frac = max(frac, 0.5) if eq >= 0.55 else frac
        else:
            frac = 0.5
        bet = pot * frac
    elif BLUFF and _bluff(c, m, eq, made, nn):
        bet = pot * (0.6 if street != "river" else 0.7)
    if bet is None:
        return check()
    if bet >= 0.5 * stack or (stack - bet) <= 0.3 * (pot + bet):
        return all_in()
    return _legal_raise(s, max(bet, c.bb))


def _bluff(c, m, eq, made, nn):
    """Capped semi-bluffs and bluffs; never into likely calling stations."""
    s = c.s
    if c.n > 2 or c.pot < 3 * c.bb or c.stack < 0.4 * c.pot:
        return False
    if MODEL and any(_is_station(m, o.seat) for o in c.opps):
        return False
    r = _RNG.random()
    if s.street != "river" and c.n == 1 and not made and 0.30 <= eq < _VALUE[nn]:
        return r < 0.50                                    # semi-bluff with a draw
    if s.street == "flop" and c.n == 1 and c.pf_aggr_me and not made and eq < _VALUE[nn]:
        return r < (0.55 if c.ip else 0.40)                # continuation bet
    if s.street == "river" and c.n == 1 and eq < 0.12 and c.ip:
        return r < 0.18                                    # occasional river bluff
    return False


# ----------------------------------------------------------------------
def _decide(state):
    t0 = time.perf_counter()
    t_end = t0 + TIME_SOFT
    if not any(p.in_hand for p in state.opponents):
        return check() if state.can_check else call()   # nobody to play against
    m = _memory(state)
    if MODEL:
        _update_model(m, state)
    c = _build_ctx(state, m)
    if state.street == "preflop":
        return _preflop(c, m, t_end)
    if not EQ_POST:
        return _legacy_post(c)
    return _postflop(c, m, t_end)


def _legacy_post(c):
    s = c.s
    return check() if s.can_check else (call() if s.pot_odds <= 0.2 else fold())


def _fallback(state):
    """Safest sensible move if anything above fails."""
    if state.can_check:
        return check()
    return call() if state.pot_odds <= 0.15 else fold()


def decide(state):
    """
    Called every time it's our turn. Returns one move: fold(), check(), call(),
    raise_to(amount) or all_in(). Never raises.
    """
    try:
        return _decide(state)
    except Exception:
        return _fallback(state)


decide_preflop = decide


# ----------------------------------------------------------------------
# You don't need to change anything below this line.
if __name__ == "__main__":
    run(decide, name=BOT_NAME)
