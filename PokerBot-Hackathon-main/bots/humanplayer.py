
from pokerbot import all_in, call, check, fold, raise_to, run

BOT_NAME = "Human"

def decide(state):
    print(f"\n--- hand {state.hand_number}/{state.hands_in_match} | {state.street} ---")
    print("you:", " ".join(state.my_cards), "| board:", " ".join(state.board) or "-")
    print(f"pot {state.pot} | to call {state.to_call} | your stack {state.my_stack}")
    print("others:", ", ".join(f"{p.name} {p.stack} {p.status}" for p in state.opponents))
    print("recent:", "; ".join(f"{m.name} {m.action} {m.amount}" for m in state.history[-8:]))
    if state.can_raise:
        print(f"raise range: {state.min_raise}-{state.max_raise}")
    while True:
        s = input("f=fold  c=check/call  r <amount>=raise to  a=all-in > ").split()
        if s and s[0] == "f":
            return fold()
        if s and s[0] == "c":
            return call()          # the client turns this into a check when it's free
        if s and s[0] == "a":
            return all_in()
        if len(s) == 2 and s[0] == "r" and s[1].isdigit():
            return raise_to(int(s[1]))
        print("?")


if __name__ == "__main__":
    run(decide, name=BOT_NAME)