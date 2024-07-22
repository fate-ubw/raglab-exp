


rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
retrieval_tokens_names = ["[No Retrieval]","[Retrieval]"]
utility_tokens_names = ["[Utility:1]", "[Utility:2]"]
ground_tokens_names = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
other_special_tokens = ["<s>", "</s>", "[PAD]",
                        "<unk>", "<paragraph>", "</paragraph>"]
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]


def load_special_tokens(tokenizer, use_grounding=False, use_utility=False):
    retrieval_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_tokens_names}
    relevant_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
        relevant_tokens[token] = tokenizer.convert_tokens_to_ids(token)


    ground_tokens = None
    if use_grounding is True:
        ground_tokens = {}
        for token in ground_tokens_names:
            ground_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    utility_tokens = None 
    if use_utility is True:
        utility_tokens = {}
        for token in utility_tokens_names:
            utility_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return retrieval_tokens, relevant_tokens, ground_tokens, utility_tokens