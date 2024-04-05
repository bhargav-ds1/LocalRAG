import llama_index.core
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from tonic_validate import ValidateScorer, ValidateApi
from tonic_validate import Benchmark
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.settings import (Settings, llm_from_settings_or_context)
from llama_index.core.llms import LLM
import json
import os
from tonic_ragas_logger import RagasValidateApi
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.integrations.llama_index import DeepEvalAnswerRelevancyEvaluator
from deepeval.models import DeepEvalBaseLLM

from Guardrails import LlamaGuardModerator


class Evaluation:
    def __init__(self, query_engine, project_id, evaluation_dataset, eval_output_dir):
        self.query_engine = query_engine
        self.project_id = project_id
        self.evaluation_dataset = evaluation_dataset
        self.eval_output_dir = eval_output_dir
        if not os.path.exists(self.eval_output_dir):
            os.makedirs(self.eval_output_dir)

    def evaluate(self):
        benchmark_df = self.localLLMEvaluate()


    def pheonixEvaluation(self):
        import phoenix as px
        from phoenix.experimental.evals import (
            HallucinationEvaluator,
            OpenAIModel,
            QAEvaluator,
            RelevanceEvaluator,
            run_evals,
        )
        from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
        from phoenix.trace import DocumentEvaluations, SpanEvaluations
        # initialize the evaluators
        eval_model = llm_from_settings_or_context(Settings, self.query_engine._retriever.get_service_context())
        hallucination_evaluator = HallucinationEvaluator(eval_model)
        qa_correctness_evaluator = QAEvaluator(eval_model)
        relevance_evaluator = RelevanceEvaluator(eval_model)
        # query the query engine with the queries from the rag dataset
        labelledRAGDataset = self.load_labelled_Rag_dataset()
        labelledRAGDataset.make_predictions_with(
            self.query_engine,
            show_progress=True,
            batch_size=1,
            sleep_time_in_seconds=1,
        )
        # get the queries and responses from the pheonix client
        queries_df = get_qa_with_reference(px.Client())
        retrieved_documents_df = get_retrieved_documents(px.Client())

        # Run evaluations using these queries and responses
        hallucination_eval_df, qa_correctness_eval_df = run_evals(
            dataframe=queries_df,
            evaluators=[hallucination_evaluator, qa_correctness_evaluator],
            provide_explanation=True,
        )
        relevance_eval_df = run_evals(
            dataframe=retrieved_documents_df,
            evaluators=[relevance_evaluator],
            provide_explanation=True,
        )[0]
        # log evaluations
        px.Client().log_evaluations(
            SpanEvaluations(eval_name="Hallucination", dataframe=hallucination_eval_df),
            SpanEvaluations(eval_name="QA Correctness", dataframe=qa_correctness_eval_df),
            DocumentEvaluations(eval_name="Relevance", dataframe=relevance_eval_df),
        )

    def deepevalEvaluation(self):
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        llm = llm_from_settings_or_context(Settings, self.query_engine._retriever.get_service_context())
        deepevalLLM = CustomEvaluationModel(llm)
        answer_relevance_metric = AnswerRelevancyMetric(model=deepevalLLM)

    def ragasEvaluation(self):
        from phoenix.trace import using_project
        import pandas as pd
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            context_precision,
            context_recall,
            faithfulness,
        )
        dataset = self.load_labelled_Rag_dataset()
        with using_project("ragas-evals"):
            evaluation_result = evaluate(
                dataset=Dataset.from_dict({'question':5}),
                metrics=[faithfulness, answer_correctness, context_recall, context_precision],
            )
        eval_scores_df = pd.DataFrame(evaluation_result.scores)
        return eval_scores_df
    def localLLMEvaluate(self):
        from Evaluation import RagEvaluatorLocal
        labelledRAGDataset = self.load_labelled_Rag_dataset()
        rag_evaluator_local = RagEvaluatorLocal(
            query_engine=self.query_engine,
            rag_dataset=labelledRAGDataset,
            judge_llm=llm_from_settings_or_context(Settings, llama_index.core.global_service_context),
            eval_output_dir=self.eval_output_dir
        )

        # PERFORM EVALUATION
        benchmark_df = rag_evaluator_local.run()  # async arun() also supported
        print(benchmark_df)
        return benchmark_df

    def tonic_evaluate(self):
        scorer = ValidateScorer()
        benchmark = self.get_benchmark_from_dataset()
        run = scorer.score(benchmark, self.get_rag_response)
        validate_api = ValidateApi(os.environ['TONIC_VALIDATE_API_KEY'])
        validate_api.upload_run(self.project_id, run)

    def get_rag_response(self, prompt):
        def get_response_from_stream(response):
            if hasattr(response, 'response') and response.response == 'Empty Response':
                return str(response.response)
            elif response.response_txt is None and response.response_gen is not None:
                response_txt = ""
                for text in response.response_gen:
                    response_txt += text
                return response_txt
            else:
                return response.response_txt

        if isinstance(self.query_engine, BaseQueryEngine):
            response = self.query_engine.query(prompt)
        elif isinstance(self.query_engine, BaseChatEngine):
            response = self.query_engine.chat(prompt)
        elif isinstance(self.query_engine, LlamaGuardModerator):
            response = self.query_engine.moderate_and_query(prompt)
        else:
            raise ValueError('Unknown query engine response type in evaluation.py')

        if len(response.source_nodes) > 0:
            context = [x.text for x in response.source_nodes]
        else:
            context = ['No context retrieved']
        return {
            "llm_answer": get_response_from_stream(response),
            "llm_context_list": context
        }

    def get_benchmark_from_dataset(self):
        labelledRAGDataset = self.load_labelled_Rag_dataset()
        questions_list = [i['query'] for i in labelledRAGDataset.dict()['examples']]
        answers_list = [i['reference_answer'] for i in labelledRAGDataset.dict()['examples']]
        benchmark = Benchmark(questions=questions_list[87:98], answers=answers_list[87:98])
        return benchmark

    def load_labelled_Rag_dataset(self):
        labelledRAGDataset = LabelledRagDataset.from_json(self.evaluation_dataset)
        return labelledRAGDataset


class CustomEvaluationModel(DeepEvalBaseLLM):
    def __init__(
            self,
            model: LLM,
            *args,
            **kwargs,
    ):
        model_name = None
        custom_model = None
        if isinstance(model, str):
            model_name = model

        elif isinstance(model, LLM):
            custom_model = model
        elif model is None:
            raise ValueError('Provide a valid LLM for evaluating the query and responses')
        self.custom_model = custom_model
        super().__init__(model_name, *args, **kwargs)

    def load_model(self):
        if self.custom_model:
            return self.custom_model

    def _call(self, prompt: str) -> str:
        llm = self.load_model()
        return llm.invoke(prompt).content
    def get_model_name(self):
        if self.custom_model:
            return self.custom_model._llm_type

        elif self.model_name:
            return self.model_name
