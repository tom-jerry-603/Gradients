def get_card_id(card_str: str) -> int:
    rank_map = {
        "A": 1, "T": 10, "J": 11, "Q": 12, "K": 13
    }

    rank_part = card_str[:-1]   # handles "10h" safely
    suit = card_str[-1]

    if rank_part.isdigit():
        rank = int(rank_part)
    else:
        rank = rank_map[rank_part]

    card_id = rank - 1
    if suit == 'c':
        card_id += 13
    elif suit == 'd':
        card_id += 26
    elif suit == 'h':
        card_id += 39

    return card_id

def get_deadwood(game_state: str, action: str | None) -> int:
    import re
    from functools import lru_cache
    # -------------------------
    # Extract cards from board
    # -------------------------
    board_section = game_state.split("Player0: Deadwood=")[-1].split("Stock size:")[0]
    cards = re.findall(r"[A2-9TJQK][cdhs]", board_section)

    card_ids = [get_card_id(c) for c in cards]

    # Remove discarded card if action is discard
    if action is not None and action.isdigit() and int(action) < 52:
        discard_id = int(action)
        if discard_id in card_ids:
            card_ids.remove(discard_id)

    # If draw upcard
    if action == "52":
        upcard_match = re.search(r"Upcard:\s*([A2-9TJQK][cdhs])", game_state)
        if upcard_match:
            card_ids.append(get_card_id(upcard_match.group(1)))

    n = len(card_ids)
    if n == 0:
        return 0

    # -------------------------
    # Precompute meld masks
    # -------------------------
    meld_masks = []

    # ---- Sets (same rank, >=3)
    rank_groups = {}
    for i, cid in enumerate(card_ids):
        rank = cid % 13
        rank_groups.setdefault(rank, []).append(i)

    for group in rank_groups.values():
        if len(group) >= 3:
            mask = 0
            for idx in group:
                mask |= (1 << idx)
            meld_masks.append(mask)

    # ---- Runs (same suit, consecutive >=3)
    suit_groups = {}
    for i, cid in enumerate(card_ids):
        suit = cid // 13
        suit_groups.setdefault(suit, []).append((cid % 13, i))

    for group in suit_groups.values():
        group.sort()
        m = len(group)
        for start in range(m):
            for end in range(start + 2, m):
                consecutive = True
                for k in range(start, end):
                    if group[k + 1][0] != group[k][0] + 1:
                        consecutive = False
                        break
                if consecutive:
                    mask = 0
                    for k in range(start, end + 1):
                        _, idx = group[k]
                        mask |= (1 << idx)
                    meld_masks.append(mask)

    # -------------------------
    # Bitmask DP
    # -------------------------
    card_values = [min(10, cid % 13 + 1) for cid in card_ids]
    total_value = sum(card_values)

    @lru_cache(None)
    def dfs(used_mask):
        best_meld_value = 0
        for meld in meld_masks:
            if (meld & used_mask) == 0:
                meld_value = sum(
                    card_values[i] for i in range(n)
                    if (meld >> i) & 1
                )
                best_meld_value = max(
                    best_meld_value,
                    meld_value + dfs(used_mask | meld)
                )
        return best_meld_value

    max_meld_value = dfs(0)

    deadwood = total_value - max_meld_value
    return deadwood

        

def get_reward(game_state: str, action: str) -> float:
    legal_action_num_list = game_state.split("Legal Actions:\n")[-1].split("\n\nYour choice")[0].strip().split("\n") 
    legal_ids = [line.split("->")[0].strip() for line in legal_action_num_list]
    if action not in legal_ids:
        return -15.0
    
    if action == "55":  # Knock action
        return 25.0
    elif action == "53" or action == "54":  # Draw from stock
        return -5.0
    
    prev_deadwood = get_deadwood(game_state, None)
    next_deadwood = get_deadwood(game_state, action)
    return prev_deadwood - next_deadwood


def rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import re
    DEBUG = False

    games_to_task_id_range = {
        "goofspiel": (0, 99999999),
        "liars_dice": (100000000, 199999999),
        "leduc_poker": (200000000, 299999999),
        "gin_rummy": (300000000, 399999999),
        "othello": (400000000, 499999999),
        "backgammon": (500000000, 599999999),
        "hex": (600000000, 699999999),
        "clobber": (700000000, 799999999),
    }

    selected_game = "gin_rummy"
    
    if not getattr(rollout_first_prompt_and_completion, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]
        if not server_list:
            base_url = ""
            if DEBUG:
                print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]
        rollout_first_prompt_and_completion.base_url = base_url
        try:
            if DEBUG:
                print(f"Initializing environment on rank {rank} at {base_url}...")
            payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts", "mcts_max_simulations": 25, "mcts_num_rollouts": 1}
            create_res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
            create_res.raise_for_status()
            rollout_first_prompt_and_completion.initialized = True
            rollout_first_prompt_and_completion.episode_id = None
            rollout_first_prompt_and_completion.total_rounds = 1
            rollout_first_prompt_and_completion.winner_rounds = 0
            rollout_first_prompt_and_completion.hint_rate = 1.0
            rollout_first_prompt_and_completion.final_10_rounds = []
            rollout_first_prompt_and_completion.messages = []
            if DEBUG:
                print(f"Environment initialized. Rank: {rank}.")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    env_endpoint = rollout_first_prompt_and_completion.base_url


    all_prompt_ids: list[list[int]] = []
    all_completion_ids: list[list[int]] = []
    all_logprobs: list[list[float]] = []
    all_rewards: list[float] = []
    action_to_sends: list[str] = []

    tokenizer = trainer.processing_class
    TIMEOUT = 2400
    hint = 'Focus on minimizing your deadwood and strategically knocking at the right time.\n'
    format_instructions = 'Your output must strictly follow this format: "Thought:\nyour thoughts ONLY in text.\n\nAction:\nONLY your action ID (a single number)."'
    if random.random() < rollout_first_prompt_and_completion.hint_rate:
        format_instructions = hint + format_instructions

    if rollout_first_prompt_and_completion.messages == []:
        try:
            game_id = random.randint(games_to_task_id_range[selected_game][0], games_to_task_id_range[selected_game][1])
            initial_payload = {"task_id": game_id, "seed": random.randint(0, 1000000), "opponent": "mcts", "mcts_max_simulations": 25, "mcts_num_rollouts": 1}
            reset_res = requests.post(f"{env_endpoint}/reset", json=initial_payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]
            rollout_first_prompt_and_completion.episode_id = result_block.get("episode_id", "")
            current_observation = result_block.get("observation", "")
            game_rule = "# Game Rules\nGIN RUMMY RULES:\n\nSETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\nMELDS (Valid Combinations):\n1. SET: 3+ cards of SAME RANK (e.g., 7:spades: 7:hearts: 7:clubs:)\n2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5:diamonds: 6:diamonds: 7:diamonds:)\nExamples:\n- Valid runs: A:spades:-2:spades:-3:spades:, 9:hearts:-10:hearts:-J:hearts:-Q:hearts:, 10:clubs:-J:clubs:-Q:clubs:-K:clubs:\n- Invalid: K:spades:-A:spades:-2:spades: (Ace is LOW only, not wraparound)\n\nCARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n- Suits: s(spades:spades:), h(hearts:hearts:), d(diamonds:diamonds:), c(clubs:clubs:)\n- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\nGAME PHASES:\n1. FirstUpcard: Choose to draw first upcard or pass (action IDs: 52=Draw upcard, 54=Pass)\n2. Draw: Choose to draw from upcard or stock pile (action IDs: 52=Draw upcard, 53=Draw stock)\n3. Discard: Choose which card to discard (action ID = card's index number, shown in Legal Actions)\n4. Layoff: After opponent knocks, add cards to their melds or pass (action IDs: card indices or 54=Pass)\n5. Knock: Declare end of hand when deadwood ≤ knock_card value\n\nEACH TURN:\n1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)\n2. DISCARD phase: Choose ONE card from hand to discard (use card's action ID from Legal Actions)\n\nKNOCKING:\n- When deadwood ≤ knock_card value (8-10), you MAY knock to end hand\n- Gin: ALL cards form melds (0 deadwood) = 25-point bonus\n\nSCORING: Winner scores difference in deadwood point values.\nCard Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\nIMPORTANT: Always respond with the action ID number ONLY, never card names.\n\nCurrent hand and game state will be provided each turn. Focus on minimizing your deadwood and strategically knocking at the right time!"
            
            initial_messages = [{"role": "system", "content": f"You are an agent playing {selected_game}.\n\n{game_rule}"}]
            initial_messages.append({"role": "user", "content": current_observation + format_instructions})
            rollout_first_prompt_and_completion.messages = initial_messages
        except Exception as e:
            print(f"CRITICAL: Failed during initial rollout on rank {rank}: {e}")
            raise e
    
    if DEBUG:
        print("\nCurrent State:\n", rollout_first_prompt_and_completion.messages[-1]["content"])
    
    for i, prompt in enumerate(prompts):
        if DEBUG:
            print(f"\n=== Starting rollout for Prompt {i+1}/{len(prompts)} ===")
        rollout_outputs = generate_rollout_completions(trainer, prompts=[rollout_first_prompt_and_completion.messages], as_chat=True)[0]
        prompt_ids = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        match = re.search(r"Action:\s*(\d+)", completion_text)
        if match:
            action_to_send = match.group(1)
        else:
            action_to_send = None

        reward = get_reward(rollout_first_prompt_and_completion.messages[-1]["content"], action_to_send)
        all_prompt_ids.append(prompt_ids)
        all_completion_ids.append(completion_ids)
        all_logprobs.append(logprobs)
        all_rewards.append(reward)
        action_to_sends.append(action_to_send)
        if DEBUG:
            print(f"Model Output Action: {action_to_send} | Reward: {reward}", flush=True)
    

    max_reward_id = max(range(len(all_rewards)), key=lambda idx: all_rewards[idx])
    best_action = action_to_sends[max_reward_id]
    if DEBUG:
        print(f"\nBest Action Selected: {best_action} with Reward: {all_rewards[max_reward_id]}", flush=True)
    done = False
    try:
        formatted_observation = ""
        step_payload = {"action": best_action, "episode_id": rollout_first_prompt_and_completion.episode_id}
        step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
        step_res.raise_for_status()
        step_data = step_res.json()
        step_block = step_data["result"]
        formatted_observation = step_block.get("observation", "") # "Game Over: gin_rummy\nYou were Player 0.\n\nFinal Returns: [44.0, -44.0]\nYour Return: 44.0\nNormalized Score: 0.679\nResult: WIN"
        done = step_block.get("done", False)
        rollout_first_prompt_and_completion.messages[1] = {"role": "user", "content": formatted_observation + format_instructions}
    except Exception as e:
        raise e

    if done:
        if "WIN" in formatted_observation:
            rollout_first_prompt_and_completion.winner_rounds += 1
            rollout_first_prompt_and_completion.hint_rate = max(0.0, rollout_first_prompt_and_completion.hint_rate - 0.05)
        else:
            rollout_first_prompt_and_completion.hint_rate = min(1.0, rollout_first_prompt_and_completion.hint_rate + 0.05)
        rollout_first_prompt_and_completion.final_10_rounds.append("WIN" if "WIN" in formatted_observation else "LOSS" if "LOSS" in formatted_observation else "DRAW")
        if len(rollout_first_prompt_and_completion.final_10_rounds) > 10:
            rollout_first_prompt_and_completion.final_10_rounds.pop(0)
        print(f"\n✅ {rollout_first_prompt_and_completion.total_rounds} Round Finished\n  \
              Result: {'WIN' if 'WIN' in formatted_observation else 'LOSS'}\n  \
              Wins: {rollout_first_prompt_and_completion.final_10_rounds}\n  \
              Win Rate: {rollout_first_prompt_and_completion.winner_rounds / rollout_first_prompt_and_completion.total_rounds:.2%}\n \
              Hint Rate: {rollout_first_prompt_and_completion.hint_rate}", flush=True)
        rollout_first_prompt_and_completion.messages = []
        rollout_first_prompt_and_completion.total_rounds += 1

    

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_rewards": all_rewards
    }

def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)

# if __name__ == "__main__":
#     game_state = "Game: gin_rummy\nYou are Player 0.\n\nCurrent State:\n\nKnock card: 10\nPrev upcard: 6s\nRepeated move: 0\nCurrent player: 0\nPhase: Draw\n\nPlayer0: Deadwood=31\n+--------------------------+\n|                          |\n|                          |\n|    3d  5d    8d          |\n|Ah2h  4h      8h          |\n+--------------------------+\n\nStock size: 37  Upcard: 6c\nDiscard pile: \n\nPlayer1:\n+--------------------------+\n|                          |\n|                          |\n|                          |\n|                          |\n+--------------------------+\n\n\nLegal Actions:\n  52 -> Player: 0 Action: Draw upcard\n  53 -> Player: 0 Action: Draw stock\n\nYour choice (action ID only):"
#     action = "52"
#     print(get_deadwood(game_state, action))