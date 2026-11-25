from pokerbot import GameState
from PleaseFold import decide

state = GameState(my_cards=["As", "Ad"], board=[], street="preflop", pot=30,
                  to_call=20, can_check=False, can_raise=True, min_raise=40,
                  max_raise=1000, my_stack=980, my_bet=0, my_seat=0,
                  opponents=[], history=[], hand_number=1, hands_in_match=20,
                  button=0)
print(decide(state))      # e.g. {'action': 'raise', 'amount': 120}