import re
import json
import pandas as pd
from openai import OpenAI
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List
from langchain_core.prompts import PromptTemplate
from weclone.data.models import QaPair, QaPairScore
from weclone.prompts.clean_data import CLEAN_PROMPT
from weclone.utils.log import logger

@dataclass
class CleaningStrategy(ABC):
    """数据清洗策略的抽象基类"""

    make_dataset_config: Dict

    @abstractmethod
    def clean(self, data: Any) -> Any:
        pass

@dataclass
class OlineLLMCleaningStrategy(CleaningStrategy):
    """使用大模型进行数据清洗的策略"""

    def judge(self, data: List[QaPair]) -> None:
        logger.info("开始使用在线模型对数据打分")

        logger.info(self.make_dataset_config.get("llm_api_key", ""))
        logger.info(self.make_dataset_config.get("base_url", ""))

        client = OpenAI(
            api_key=self.make_dataset_config.get("llm_api_key", ""),
            base_url=self.make_dataset_config.get("base_url", ""),
        )
        prompt_template = PromptTemplate.from_template(CLEAN_PROMPT)

        parsed_scores = []
        for qa in data:
            prompt_text = prompt_template.invoke({"id": qa.id, "Q": qa.instruction, "A": qa.output}).text
            try:
                response = client.chat.completions.create(
                    model=self.make_dataset_config.get("model_name", "deepseek-chat"),
                    messages=[
                        {"role": "system", "content": self.make_dataset_config.get("default_system", "")},
                        {"role": "user", "content": prompt_text},
                    ],
                    stream=False,
                )
                result_text = response.choices[0].message.content
                # 去掉开头和结尾的 ```json 或 ``` 等代码块标记
                result_text = re.sub(r"^```json\s*|```$", "", result_text.strip(), flags=re.MULTILINE)
                print(result_text)
                score_data = json.loads(result_text)
                qa_score = QaPairScore(**score_data)
                parsed_scores.append(qa_score)
            except Exception as e:
                logger.error(f"调用在线模型或解析结果失败，QA ID {qa.id}: {str(e)}")

        score_map = {score.id: score.score for score in parsed_scores}
        for qa in data:
            if qa.id in score_map:
                qa.score = score_map[qa.id]
            else:
                logger.warning(f"未获取到QA ID {qa.id}的分数，默认赋值0")
                qa.score = 0

        # 统计分数分布，打印日志（和本地版本保持一致）
        scores = [qa.score for qa in data if qa.score is not None]
        score_series = pd.Series(scores)
        score_counts = score_series.value_counts().sort_index()
        score_percentages = score_series.value_counts(normalize=True).sort_index() * 100
        pd.set_option("display.unicode.east_asian_width", True)
        distribution_df = pd.DataFrame({
            "数量": score_counts,
            "占比(%)": score_percentages.round(2),
        })
        distribution_df.index.name = "分数"
        printable_df_str = distribution_df.reset_index().to_string(index=False)
        logger.success(f"在线模型打分分数分布情况:\n{printable_df_str}")

    def clean(self, data: List[QaPair]) -> List[QaPair]:
        """
        根据打分结果，删除分数低于阈值的数据。
        """
        threshold = self.make_dataset_config.get("clean_dataset", {}).get("llm", {}).get("accept_score", 1)
        return [
            qa
            for qa in data
            if qa.score is not None and qa.score >= threshold
        ]
