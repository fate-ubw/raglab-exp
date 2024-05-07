INSTRUCTION_LAB = [
    {
        "rag_name": "-------------------Template-------------------------",
        "dataset_name": ""
    },
    {
        "rag_name": "Rules of naming:'-' seperate for naming. For example: Algorithm_name-mode-specific_stage",
        "dataset_name": "dataset name",
        "instruction": "Fill in your instruction here"
    },
    {
        "rag_name": "-------------------Naive Rag-------------------------",
        "dataset_name": ""
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "",
        "instruction": "### Instruction:\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "ArcChallenge",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "ArcChallenge",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n Determine the claim based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Naive_rag-without_retrieval",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "-------------------Query Rewrite Rag-------------------------",
        "dataset_name": ""
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "PopQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "TriviaQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "StrategyQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited.### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "HotPotQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "MMLU",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "ArcChallenge",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "ArcChallenge",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "PubHealth",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "ASQA",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "Factscore",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "query_rewrite_rag-rewrite",
        "dataset_name": "Feverous",
        "instruction": "Provide a better search query for Wikipedia to answer the given question, end the query with '**'. \n\n Question: Ezzard Charles was a world champion in which sport? \n\n Query: Ezzard Charles champion** \n\n Question: What is the correct name of laughing gas? \n\n Query: laughing gas name** \n\n Question: {query} \n\n Query: "
    },
    {
        "rag_name": "query_rewrite_rag-read",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n Determine the claim based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "-------------------Iter-Retgen-------------------------",
        "dataset_name": ""
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "ArcChallenge",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "Iterative_rag-read",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n Determine the claim based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "-------------------Active Rag-------------------------",
        "dataset_name": ""
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "ArcChallenge",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "active_rag-read",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n Determine the claim based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "-------------------Self Ask-------------------------",
        "dataset_name": ""
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "PopQA",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "TriviaQA",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "StrategyQA",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "HotPotQA",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\n Given four answer candidates, A, B, C and D, choose the best answer choice. \n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "MMLU",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "ArcChallenge",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "ArcChallenge",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "PubHealth",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "ASQA",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "Factscore",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "self_ask-read",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n## Input:\n\n{query}\n\n Determine the claim based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "rag_name": "self_ask-followup_question",
        "dataset_name": "Feverous",
        "instruction": "Question: Are both the directors of Jaws and Casino Royale from the same country? \n Are follow up questions needed here: Yes. Follow up: Who is the director of Jaws? \n Intermediate Answer: The director of Jaws is Steven Spielberg. \n Follow up: Where is Steven Spielberg from? \n Intermediate Answer: The United States. \n Follow up: Who is the director of Casino Royale? \n Intermediate Answer: The director of Casino Royale is Martin Campbell. \n Follow up: Where is Martin Campbell from? \n Intermediate Answer: New Zealand. \n  \nSo the final answer is: No \n{query} Are follow up questions needed here:"
    },
    {
        "rag_name": "-------------------Self Rag Reproduction-------------------------",
        "dataset_name": ""
    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"
    },

    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"

    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\nYou are only allowed to answer True or False, and generating other types of responses is prohibited.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "MMLU",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "ArcChallenge",
        "instruction": "### Instruction:\nGiven four answer candidates, A, B, C and D, choose the best answer choice.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "ASQA",
        "instruction": "### Instruction:\nAnswer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.\n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "Factscore",
        "instruction": "### Instruction:\n{query}\n\n### Response:\n"
    },
    {
        "rag_name": "selfrag_reproduction-read",
        "dataset_name": "Feverous",
        "instruction": "### Instruction:\n Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. You can only answer SUPPORTS, REFUTES, or NOT ENOUGH INFORMATION, and you are prohibited from generating other responses, and the answers generated must be all in capital letters. \n\n## Input:\n\n{query}\n\n### Response:\n"
    }
]