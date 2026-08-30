from typing import Union
import numpy as np
import copy

def apply_chat_template(
        tokenizer, 
        messages: Union[np.ndarray, str], 
        response: str,
        with_thinking=True, 
        **kwargs
    ):
    """
    Apply a chat template to given messages.
    ================================
    Args:
        - tokenizer: The tokenizer to use.
        - messages (Union[np.ndarray, str]): The messages to apply the template to.
            - str: The messages are already formatted.
            - np.ndarray: The messages are not formatted.
        - with_thinking: Whether the assistant needs to output think tags.
            - e.g., <|im_start|>assistant\n --> <|im_start|>assistant\n<think>, assistant needs to output </think> tags.
            - assistant message is added <think></think> at the beginning
        - kwargs: Additional keyword arguments to pass to the tokenizer.apply_chat_template method.
    ================================
    Returns:
        The formatted messages (str).
    """
    if isinstance(messages, str):
        return messages
    assert isinstance(messages, np.ndarray), "The messages must be a numpy array."
    messages = copy.deepcopy(messages.tolist())
    assert messages[-1]['role'] == 'user', "The last message must be a user message."
    # if not with_thinking:
    prompt_chat_str = tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
    response_chat_str = response
    return prompt_chat_str, response_chat_str
    # if with_thinking:
    #     for msg in messages:
    #         if msg['role'] == 'assistant':
    #             msg['content'] = f"<think></think>{msg['content']}"
    #     prompt_chat_str = tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
    #     prompt_chat_str = f"{prompt_chat_str}<think>"
    #     response_chat_str = f"</think>{response}"
    # return prompt_chat_str, response_chat_str
