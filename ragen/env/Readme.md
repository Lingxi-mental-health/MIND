# Environment

NOTE if you are using gymnasium, ensure that in `reset` you should call **gym.Env.reset(self, seed=seed)** and use **self.np_random** for randomness.

## BaseEnv
Methods already implemented:
- `execute_predictions`: For all environments, `execute_predictions` is the actual `step` function in the environment, will be called while rolling out the trajectory.

Methods must be implemented:
- `postprocess_predictions`: postprocess the predictions from llm into actions and validity flags.
    - e.g., extract answer from <answer> tags, or directly return the action and transform to action space (include invalid action)
- `parse_update_info_to_obs`: parse the update information into observation string.
    - update_info: (observation, reward, done, info)
    - action_is_valid: whether the action is valid
    - output: observation string to LLM
- `reset`: reset the environment.
- `step`: step with the action.
- `success`: check if the current state is successful.
- `render`: render the environment.
- `copy`: copy the environment.

Class attributes:
 - `INVALID_ACTION`: an action to handle invalid input (0 as default), used when the action is invalid

## BaseGridEnv
Environment for grid-based environments and discrete action spaces, like FrozenLakeEnv and SokobanEnv.

Methods already implemented:
- `execute_predictions`
- `parse_update_info_to_obs`
- `postprocess_predictions`
- `get_all_actions`

Methods must be implemented:
- `extract_action`: extract action from text, e.g., text is the content between `<answer>` tags
- `reset`: reset the environment.
- `step`: step with the action.
- `success`: check if the current state is successful.
- `render`: render the environment.
- `copy`: copy the environment.

Class attributes:
 - `INVALID_ACTION`: an action to handle invalid input (0 as default), used when the action is invalid
 - `ACTION_SPACE`: the action space of the environment (exclude invalid action)
 - `ACTION_LOOKUP`: a dictionary mapping actions to strings, used for rendering
 - `GRID_LOOKUP`: a dictionary mapping grid positions to strings, used for rendering
