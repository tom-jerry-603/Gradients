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
    card_values = [cid % 13 + 1 for cid in card_ids]
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

        

def get_reward(phase:str, game_state: str, action: str) -> float:
    legal_action_num_list = game_state.split("Legal Actions:\n")[-1].split("\n\nYour choice")[0].strip().split("\n") 
    legal_ids = [line.split("->")[0].strip() for line in legal_action_num_list]
    if action not in legal_ids:
        return -1.0
    
    if phase == "Discard":
        if action == '55':
            return 100.0
        prev_deadwood = get_deadwood(game_state, None)
        next_deadwood = get_deadwood(game_state, action)
        reward = prev_deadwood - next_deadwood
        return reward
    elif phase == "Draw" or phase == "FirstUpcard":
        if action == "52":
            prev_deadwood = get_deadwood(game_state, None)
            next_deadwood = get_deadwood(game_state, action)
            if next_deadwood < prev_deadwood:
                return 1.0
            elif next_deadwood < prev_deadwood + 5:
                return 0.5
            else:
                return 0.0
        else:
            return 0.3  
    elif phase == "Knock":
        prev_deadwood = get_deadwood(game_state, None)
        next_deadwood = get_deadwood(game_state, action)
        return prev_deadwood - next_deadwood


def rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import copy
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
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]
        rollout_first_prompt_and_completion.base_url = base_url
        try:
            print(f"Initializing environment on rank {rank} at {base_url}...")
            payload = {"task_id": games_to_task_id_range[selected_game][0], "seed": 42, "opponent": "mcts", "mcts_max_simulations": 25, "mcts_num_rollouts": 1}
            create_res = requests.post(f"{base_url}/reset", json=payload, timeout=300)
            create_res.raise_for_status()
            rollout_first_prompt_and_completion.initialized = True
            print(f"Environment initialized. Rank: {rank}.")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    env_endpoint = rollout_first_prompt_and_completion.base_url


    all_prompt_ids: list[list[int]] = []
    all_completion_ids: list[list[int]] = []
    all_logprobs: list[list[float]] = []
    all_rewards: list[float] = []

    tokenizer = trainer.processing_class
    TIMEOUT = 2400
    num_generations = getattr(trainer.args, "num_generations", 4)
    PHASE_TYPES = ["Discard", "Draw", "FirstUpcard", "Knock"]
    PHASE_WEIGHTS = [0.70, 0.15, 0.05, 0.10]
    

    for i, prompt in enumerate(prompts):
        
        done = False
        game_id = random.randint(games_to_task_id_range[selected_game][0], games_to_task_id_range[selected_game][1])
        phase = random.choices(PHASE_TYPES, weights=PHASE_WEIGHTS)[0]
        turn_number = 0
        
        payload = {"task_id": game_id, "seed": 42, "opponent": "mcts", "mcts_max_simulations": 25, "mcts_num_rollouts": 1}
        
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]

            episode_id = result_block.get("episode_id", "")

            current_observation = result_block.get("observation", "")
            format_instructions = 'Your output must strictly follow this format: "Thought:\nyour thoughts ONLY in text.\n\nAction:\nONLY your action ID (a single number)."'

            if DEBUG:
                print(f"Env Reset. Observation: {current_observation}", flush=True)

        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            continue
        
        selected_phase_messages = []
        messages = [{"role": "system", "content": f"You are an agent playing {selected_game}. {format_instructions}"}]
        
        messages.append({"role": "user", "content": current_observation})

        while not done and (turn_number < max_turns):
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
            if phase in messages[-1]["content"]:
                selected_phase_messages.append(copy.deepcopy(messages))

            completion_text = tokenizer.decode(rollout_outputs.get("completion_ids", []), skip_special_tokens=True).strip()

            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

            if DEBUG:
                print(f"Sending Action to Env: {action_to_send}", flush=True)

            try:
                formatted_observation = ""
                step_payload = {"action": action_to_send, "episode_id": episode_id}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                if DEBUG:
                    print(f"Env Step Response: {step_data}", flush=True)

                step_state = step_block.get("observation", "")
                done = step_block.get("done", False)
                formatted_observation = step_state
                
            except Exception as e:
                if DEBUG: 
                    print(f"Step failed: {e}")
                formatted_observation = "Invalid Action.\n\n" + formatted_observation + "\n\nPlease provide a valid action following the format instructions."
                done = False

            messages.append({"role": "assistant", "content": completion_text})
            if not done:
                messages.append({"role": "user", "content": formatted_observation})

            turn_number += 1
        
        selected_phase_message = random.choice(selected_phase_messages) if selected_phase_messages else messages
        turn_prompt_ids: list[list[int]] = []
        turn_completion_ids: list[list[int]] = []
        turn_logprobs: list[list[float]] = []
        turn_rewards: list[float] = []
        for _ in range(num_generations):
            rollout_outputs = generate_rollout_completions(trainer, prompts=[selected_phase_message], as_chat=True)[0]
            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            logprobs = rollout_outputs.get("logprobs", [])
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()

            if DEBUG:
                print(f"Sending Action to Env: {action_to_send}", flush=True)

            reward = get_reward(phase, selected_phase_message[-1]["content"], action_to_send)

            turn_prompt_ids.append(prompt_ids)
            turn_completion_ids.append(completion_ids)
            turn_logprobs.append(logprobs)
            turn_rewards.append(reward)
        
        all_prompt_ids.extend(turn_prompt_ids)
        all_completion_ids.extend(turn_completion_ids)
        all_logprobs.extend(turn_logprobs)
        all_rewards.extend(turn_rewards)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_rewards": all_rewards
    }

def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)

if __name__ == "__main__":
    game_state = "Game: gin_rummy\nYou are Player 0.\n\nCurrent State:\n\nKnock card: 10\nPrev upcard: 6s\nRepeated move: 0\nCurrent player: 0\nPhase: Draw\n\nPlayer0: Deadwood=31\n+--------------------------+\n|                          |\n|                          |\n|    3d  5d    8d          |\n|Ah2h  4h      8h          |\n+--------------------------+\n\nStock size: 37  Upcard: 6c\nDiscard pile: \n\nPlayer1:\n+--------------------------+\n|                          |\n|                          |\n|                          |\n|                          |\n+--------------------------+\n\n\nLegal Actions:\n  52 -> Player: 0 Action: Draw upcard\n  53 -> Player: 0 Action: Draw stock\n\nYour choice (action ID only):"
    action = "52"
    print(get_deadwood(game_state, action))