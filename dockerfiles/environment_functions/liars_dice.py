import re
import math
def parse_liars_dice_state(text):
    result = {}

    # ---- total dice ----
    m = re.search(r"Total dice in game:\s*(\d+)", text)
    result["total_dice"] = int(m.group(1)) if m else None

    # ---- my dice ----
    m = re.search(r"Your dice:\s*\[([0-9,\s]+)\]", text)
    if m:
        result["my_dice"] = [int(x.strip()) for x in m.group(1).split(",")]
    else:
        result["my_dice"] = []

    # ---- current bid ----
    if "No bid yet" in text:
        result["current_bid"] = None
    else:
        m = re.search(r'Current bid:\s*"(\d+)-(\d+)"', text)
        if m:
            result["current_bid"] = {
                "quantity": int(m.group(1)),
                "face": int(m.group(2))
            }
        else:
            result["current_bid"] = None

    # ---- legal actions ----
    actions = {}
    for match in re.finditer(r"\s*(\d+)\s*->\s*([0-9\-A-Za-z]+)", text):
        action_id = int(match.group(1))
        action_value = match.group(2)
        actions[action_id] = action_value

    result["legal_actions"] = actions

    return result


def binom_prob_at_least(n, k, p):
    """P(X >= k) for X ~ Binomial(n,p)"""
    prob = 0.0
    for i in range(k, n+1):
        prob += math.comb(n, i) * (p**i) * ((1-p)**(n-i))
    return prob


def get_reward(rollout_first_prompt_and_completion, action):
    action = int(action)

    current_bid = rollout_first_prompt_and_completion.current_bid
    legal_actions = rollout_first_prompt_and_completion.legal_actions
    my_dice = rollout_first_prompt_and_completion.current_dice
    total_dice = rollout_first_prompt_and_completion.total_dice_num

    # illegal move
    if action not in legal_actions:
        return -1.0

    quantity = current_bid["quantity"]
    face = current_bid["face"]

    my_count = 0
    for d in my_dice:
        if face == 6:
            if d == 6:
                my_count += 1
        else:
            if d == face or d == 6:
                my_count += 1

    other_dice = total_dice - len(my_dice)
    need_from_others = max(0, quantity - my_count)

    # probability a die matches
    if face == 6:
        p = 1/6
    else:
        p = 1/3

    prob_bid_true = binom_prob_at_least(other_dice, need_from_others, p)

    # ---------- action: say liar ----------
    if action == 60:
        reward = 1 - prob_bid_true
        return reward * 2 - 1  # normalize to [-1,1]

    # ---------- action: make bid ----------
    bid_str = legal_actions[action]
    q, f = map(int, bid_str.split("-"))

    my_count_new = 0
    for d in my_dice:
        if f == 6:
            if d == 6:
                my_count_new += 1
        else:
            if d == f or d == 6:
                my_count_new += 1

    need_new = max(0, q - my_count_new)

    if f == 6:
        p_new = 1/6
    else:
        p_new = 1/3

    prob_new_bid_true = binom_prob_at_least(other_dice, need_new, p_new)

    reward = prob_new_bid_true

    return reward * 2 - 1

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

    selected_game = "liars_dice"
    
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
    hint = 'Base your bids on the number of dice you see, assuming roughly one-third of the unknown dice (plus your own) match, and always increase the total count or face value while watching for bluffs.\n'
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

            dt = parse_liars_dice_state(result_block['observation'])
            rollout_first_prompt_and_completion.current_dice = dt['my_dice']
            rollout_first_prompt_and_completion.total_dice_num = dt['total_dice']
            rollout_first_prompt_and_completion.current_bid = None

            current_observation = result_block.get("observation", "")
            game_rule = "# Game Rules\nLIAR'S DICE RULES:\n\nSetup: Each player has N dice (1-5 depending on variant). All players roll their dice secretly.\n\nGoal: Make bids about total dice across ALL players, or call \"Liar\" on opponent's bid.\n\nActions:\n- Bid (quantity, face): Claim there are at least 'quantity' dice showing 'face' among all dice.\n- Call Liar: Challenge the previous bid.\n\nBidding rules: Each bid must be higher than the previous bid. \"Higher\" means:\n  - Same face value but higher quantity (e.g., \"2 fours\" beats \"1 four\")\n  - Same quantity but higher face value (e.g., \"2 fives\" beats \"2 fours\")\n\nWild dice: 6s are WILD and count as ANY face value.\n- When counting dice for a bid, include 6s in the count\n- Example: Bid \"3 fours\" means at least 3 dice showing EITHER 4 OR 6\n\nWinning: If you call Liar and previous bid was false, opponent loses. If bid was true or exact, you lose.\n\n\n"
            
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
        rollout_first_prompt_and_completion.current_bid = parse_liars_dice_state(rollout_first_prompt_and_completion.messages[-1]["content"])['current_bid']
        rollout_first_prompt_and_completion.legal_actions = parse_liars_dice_state(rollout_first_prompt_and_completion.messages[-1]["content"])['legal_actions']
        reward = get_reward(rollout_first_prompt_and_completion, action_to_send)
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
