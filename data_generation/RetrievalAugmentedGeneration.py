from typing import List, Dict, Union, Any, Optional
from llm_multiprocessing_inference import get_answers
import pandas as pd
import torch
import json
import os


question_answering_retrieval_system_prompt = """
I am going to ask you a question and you will answer using only the information provided to you. %s

The context is in the form of a JSON dictionary where the keys are the ids of the extracts and the values are the text of the extracts. The extract’s language is English, French or Spanish.

Return the answer in the form of a JSON dictionary with the following elements:
"answer": the answer to the question. Return an empty string ("-") if the answer is not available in the context. When you report on challenges, complement your analysis as much as possible with numerical data from the text. The answer has to be in %s.
"relevance": a score from 0 to 1 indicating how relevant the answer is to the question. Scores under 0.5 mean that the answer is not relevant. The score of an answer that doesn't answer any part of the question is 0 and the score of an answer that answers all parts of the question is 1. Partially relevant answers should be scored between 0.5 and 1. It is better to return unnecessary information than missing important ones.
"evidence": a list of the ids of the extracts that were used to answer the question. Extracts with similar information should all be included in the list.
If the context does not contain enough information to answer the question, return an empty dictionary. Base your answer only on the extracts in the context and do not add any new information.
Make the answer complete, precise, self-contained and as detailed as possible. Do not make any reference to the extract ID in the answer. If different extracts answer different parts of the question, return all of them. Only provide answers from the input text and nothing else.
If no information is relevant to answer the question, return an empty dictionary. No matter the answer, do not deviate from the specified format.

### Example output:
{
    "answer": "The answer to the question.",
    "relevance": 0.8,
    "evidence": [0, 1, 3]
}

### Question(s):
%s
"""

df_relevant_columns = [
    "Extraction Text",
    "Document Title",
    "Document Publishing Date",
    "Document URL",
    "Document Source",
]

default_response = {"answer": "-", "relevance": 0.0, "evidence": []}


def _get_embeddings_similarity(
    qa_df: pd.DataFrame,
    extracts_embeddings: torch.Tensor,
    one_question_embeddings: torch.Tensor,
    n_kept_entries: int,
):
    embedding_similarity = (
        torch.matmul(extracts_embeddings.float(), one_question_embeddings.T.float())
        .reshape(-1)
        .squeeze()
    )

    # descending argsort to get the most similar embeddings
    most_similar_indices = torch.argsort(embedding_similarity, descending=True)
    most_similar_indices = most_similar_indices[:n_kept_entries]
    most_relevant_df = qa_df.iloc[most_similar_indices].copy()
    return most_relevant_df


def _get_zero_shot_reranking(
    most_relevant_df: pd.DataFrame,
    one_question: str,
    text_col: str,
    zero_shot_reranking_pipeline: Dict[str, Any],
):
    from zero_shot_classification import MultiStepZeroShotClassifier

    classifier = MultiStepZeroShotClassifier(**zero_shot_reranking_pipeline)
    tags = [one_question]
    outputs: List[Dict[str, float]] = classifier(
        entries=most_relevant_df[text_col].tolist(), tags=tags
    )
    most_relevant_df["relevance"] = [output[one_question] for output in outputs]
    most_relevant_df = most_relevant_df.sort_values(by="relevance", ascending=False)
    return most_relevant_df


def generate_one_llm_input(
    most_relevant_df,
    n_kept_entries,
    one_question: str = "",
    text_col: str = "Extraction Text",
    additional_context: str = "",
    output_language: str = "english",
):
    context = json.dumps(
        {
            i: one_info[text_col]
            for i, (_, one_info) in enumerate(
                most_relevant_df.iloc[:n_kept_entries].iterrows()
            )
        }
    )

    prompt_one_entry = [
        {
            "role": "system",
            "content": question_answering_retrieval_system_prompt
            % (additional_context, output_language, one_question),
        },
        {
            "role": "user",
            "content": context,
        },
    ]
    return prompt_one_entry


def generate_context_and_prompts(
    qa_df: pd.DataFrame,
    question_embeddings: Dict[str, torch.Tensor],
    n_kept_entries: int,
    n_initial_kept_entries: int = 200,
    zero_shot_reranking_pipeline: Optional[Dict[str, Any]] = None,
    additional_context: str = "",
    embeddings_column: str = "Embeddings",
    output_language: str = "english",
    question_answering_retrieval_system_prompt=question_answering_retrieval_system_prompt,
    text_col="Extraction Text",
):
    # st.markdown(f"Number of unique extracts: {len(qa_df)}")
    extracts_embeddings = torch.tensor(
        qa_df[embeddings_column].tolist(), dtype=torch.float16
    )

    prompts = []
    context_df = pd.DataFrame()
    for question_id, (one_question, one_question_embeddings) in enumerate(
        question_embeddings.items()
    ):

        if len(qa_df) > 1:
            if zero_shot_reranking_pipeline is not None:
                initial_kept_entries = n_initial_kept_entries
            else:
                initial_kept_entries = n_kept_entries
            most_relevant_df = _get_embeddings_similarity(
                qa_df,
                extracts_embeddings,
                one_question_embeddings,
                initial_kept_entries,
            )
            if zero_shot_reranking_pipeline is not None:
                most_relevant_df = _get_zero_shot_reranking(
                    most_relevant_df,
                    one_question,
                    text_col,
                    zero_shot_reranking_pipeline,
                )
        else:
            most_relevant_df = qa_df.copy()
        # st.dataframe(most_relevant_df)
        prompt_one_entry = generate_one_llm_input(
            most_relevant_df,
            n_kept_entries,
            one_question,
            text_col,
            additional_context,
            output_language,
        )
        prompts.append(prompt_one_entry)
        most_relevant_df["question_id"] = question_id
        context_df = pd.concat([context_df, most_relevant_df])

    return prompts, context_df


def postprocess_RAG_answers(
    answers: List[Dict[str, Union[str, float]]],
    context_df: pd.DataFrame,
    df_relevant_columns: List[str] = df_relevant_columns,
):
    final_data = []
    
    default_results = {
        "final_answer": "-",
        "final_relevance": 0.0,
        "final_context": [],
    }
    # print(answers)
    for question_id, one_answer in enumerate(answers):

        try:
            if len(one_answer) > 0 and one_answer["answer"] != "-":

                final_answer_one_question = one_answer["answer"]
                final_relevance_one_question = one_answer["relevance"]
                context_df_one_question = context_df[
                    context_df["question_id"] == question_id
                ]
                context_df_one_question = context_df_one_question.iloc[
                    [int(id) for id in one_answer["evidence"]]
                ]
                final_context_one_question = []
                for i, row in context_df_one_question.iterrows():
                    final_context_one_question.append(
                        {col: row[col] for col in df_relevant_columns}
                    )
                one_entry_final_results = {
                    "final_answer": final_answer_one_question,
                    "final_relevance": final_relevance_one_question,
                    "final_context": final_context_one_question,
                }
            else:
                one_entry_final_results = default_results
        except:
            one_entry_final_results = default_results
            print(f"Error for answer {one_answer}")
        final_data.append(one_entry_final_results)
    return final_data


def RAG(
    df: pd.DataFrame,
    question_embeddings: Dict[str, torch.Tensor],
    n_kept_entries: int,  # extracts: List[str], input_question: str
    show_progress_bar: bool = False,
    additional_context: str = "",
    embeddings_column: str = "Embeddings",
    output_language: str = "english",
    question_answering_retrieval_system_prompt=question_answering_retrieval_system_prompt,
    text_col="Extraction Text",
    api_key=os.getenv("openai_api_key"),
    api_pipeline="OpenAI",
    model="gpt-4o-mini",
    df_relevant_columns=df_relevant_columns,
) -> List[Dict[str, Union[str, float]]]:
    """
    Retrieve the question and answer information from a list of extracts using the input question.
    """

    prompts, context_df = generate_context_and_prompts(
        qa_df=df,
        question_embeddings=question_embeddings,
        n_kept_entries=n_kept_entries,
        additional_context=additional_context,
        embeddings_column=embeddings_column,
        output_language=output_language,
        question_answering_retrieval_system_prompt=question_answering_retrieval_system_prompt,
        text_col=text_col,
    )

    answers = get_answers(
        prompts=prompts,
        response_type="structured",
        model=model,
        default_response=default_response,
        api_pipeline=api_pipeline,
        api_key=api_key,
        show_progress_bar=show_progress_bar,
        additional_progress_bar_description="RAG answers generation",
    )

    final_data = postprocess_RAG_answers(
        answers=answers,
        context_df=context_df,
        df_relevant_columns=df_relevant_columns,
    )
    return final_data
