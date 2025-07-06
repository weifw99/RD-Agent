import json
from abc import abstractmethod
from pathlib import Path
from typing import Tuple

from jinja2 import Environment, StrictUndefined

from rdagent.core.experiment import Experiment
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import (
    Hypothesis,
    Hypothesis2Experiment,
    HypothesisGen,
    Scenario,
    Trace,
)
from rdagent.oai.llm_utils import APIBackend

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class LLMHypothesisGen(HypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)

    # The following methods are scenario related so they should be implemented in the subclass
    @abstractmethod
    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str) -> Hypothesis: ...

    def gen(self, trace: Trace) -> Hypothesis:
        context_dict, json_flag = self.prepare_context(trace)

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_gen"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                hypothesis_output_format=context_dict["hypothesis_output_format"],
                hypothesis_specification=context_dict["hypothesis_specification"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_gen"]["user_prompt"])
            .render(
                targets=self.targets,
                hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                RAG=context_dict["RAG"],
            )
        )

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)

        hypothesis = self.convert_response(resp)

        return hypothesis


class FactorHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "factors"


class ModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "model tuning"


class FactorAndModelHypothesisGen(LLMHypothesisGen):
    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.targets = "feature engineering and model building"


class LLMHypothesis2Experiment(Hypothesis2Experiment[Experiment]):
    @abstractmethod
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict, bool]: ...

    @abstractmethod
    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> Experiment: ...

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

    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        context, json_flag = self.prepare_context(hypothesis, trace)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=trace.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                experiment_output_format=context["experiment_output_format"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=self.targets,
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                target_list=context["target_list"],
                RAG=context["RAG"],
            )
        )

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)

        return self.convert_response(resp, hypothesis, trace)


class FactorHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "factors"


class ModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "model tuning"


class FactorAndModelHypothesis2Experiment(LLMHypothesis2Experiment):
    def __init__(self):
        super().__init__()
        self.targets = "feature engineering and model building"
