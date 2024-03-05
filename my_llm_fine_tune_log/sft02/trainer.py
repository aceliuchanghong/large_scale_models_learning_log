import transformers
from typing import Dict
from dataclasses import dataclass, field
import torch

from ChatGLM_model_engine import ChatGLMEngine
from chat_data_module import ChatDataModule


class ChatTrainer:
    def __init__(self, training_config: dataclass):
        self.training_config = training_config
        self.model_engine : ChatGLMEngine = ChatGLMEngine(training_config.model_name, training_config=training_config)
        self.data_path = training_config.data_path

    def train(self):
        self.model_engine.prepare_model_for_training()

        self.data_module = ChatDataModule(
            tokenizer=self.model_engine.tokenizer,
            data_path_train=self.data_path,
            max_tokens=self.model_engine.config.max_tokens
        )

        self.model_engine.train(data_module=self.data_module)
        