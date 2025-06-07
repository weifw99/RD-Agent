import json
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelTask
from rdagent.components.proposal import ModelHypothesis2Experiment, ModelHypothesisGen
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")

QlibModelHypothesis = Hypothesis


class QlibModelHypothesisGen(ModelHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": "In Quantitative Finance, market data could be time-series, and GRU model/LSTM model are suitable for them. Do not generate GNN model as for now.",
            "hypothesis_output_format": prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": prompt_dict["model_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = self.hypothesis_response_parse(response)
        print(f"{'##' * 10} QlibModelHypothesisGen.convert_response：response_dict: {response_dict}")
        hypothesis = QlibModelHypothesis(
            hypothesis=response_dict["hypothesis"],
            reason=response_dict["reason"],
            concise_reason=response_dict["concise_reason"],
            concise_observation=response_dict["concise_observation"],
            concise_justification=response_dict["concise_justification"],
            concise_knowledge=response_dict["concise_knowledge"],
        )
        return hypothesis

    def hypothesis_response_parse(self, response):
        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError:
            print(
                f"{'##' * 10} QlibModelHypothesisGen.convert_response：JSON 解析错误，将反斜杠替换为四个反斜杠(LaTeX 公式处理), json response: {response}")

            # 处理 JSON 解析错误，将反斜杠替换为四个反斜杠(LaTeX 公式处理)
            def escape_latex_for_json(s):
                import re
                # 将单个反斜杠替换为四个反斜杠（适用于 LaTeX 公式在 JSON 中的表示）
                return re.sub(r'\\', r'\\\\', s)

            response = escape_latex_for_json(response)
            import re

            def safe_json_loads(response: str):
                # 仅替换 JSON 值中的 LaTeX 字符串中的反斜杠（不会影响 JSON key）
                def escape_latex(match):
                    return match.group(0).replace("\\", "\\\\")

                # 替换所有 value 中包含 \ 的字段，避免误伤 key
                response = re.sub(r'(?<="formulation":\s?")[^"]+', escape_latex, response)
                response = re.sub(r'(?<="description":\s?")[^"]+', escape_latex, response)
                response = re.sub(r'(?<=".*?":\s?")[^"]+(?=")', escape_latex, response)

                return json.loads(response)

            response_dict = safe_json_loads(response)
        return response_dict


class QlibModelHypothesis2Experiment(ModelHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = prompt_dict["model_experiment_output_format"]

        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        experiment_list: List[ModelExperiment] = [t[0] for t in trace.hist]

        model_list = []
        for experiment in experiment_list:
            model_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": model_list,
            "RAG": None,
        }, True

    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> ModelExperiment:
        response_dict = json.loads(response)
        tasks = []
        for model_name in response_dict:
            description = response_dict[model_name]["description"]
            formulation = response_dict[model_name]["formulation"]
            architecture = response_dict[model_name]["architecture"]
            variables = response_dict[model_name]["variables"]
            hyperparameters = response_dict[model_name]["hyperparameters"]
            model_type = response_dict[model_name]["model_type"]
            tasks.append(
                ModelTask(
                    name=model_name,
                    description=description,
                    formulation=formulation,
                    architecture=architecture,
                    variables=variables,
                    hyperparameters=hyperparameters,
                    model_type=model_type,
                )
            )
        exp = QlibModelExperiment(tasks, hypothesis=hypothesis)
        exp.based_experiments = [t[0] for t in trace.hist if t[1]]
        return exp
