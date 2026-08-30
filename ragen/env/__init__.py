from .frozen_lake.env import FrozenLakeEnv
from .sokoban.env import SokobanEnv
from .bandit.env import BanditEnv
from .bandit.env import TwoArmedBanditEnv
from .countdown.env import CountdownEnv
from .med_dialogue.env import MedicalConsultationEnv
from .base import BaseEnv
from .med_dialogue.env_patient_llm import MedicalConsultationEnvWithPatientLLM
from .med_dialogue.env_patient_llm_rm import MedicalConsultationEnvWithPatientLLMandRM
# 明确使用 env_patient_llm_category.py（有 RAG 功能 + 完整监控指标）
# 不要切换到 env_patient_llm_category_full_reward.py 或 env_patient_llm_category_simple.py
from .med_dialogue.env_patient_llm_category import MedicalConsultationEnvWithPatientLLMCategory

__all__ = [
    'FrozenLakeEnv', 
    'SokobanEnv', 
    'BanditEnv', 
    'TwoArmedBanditEnv', 
    'CountdownEnv', 
    'BaseEnv', 
    'MedicalConsultationEnv', 
    'MedicalConsultationEnvWithPatientLLM', 
    'MedicalConsultationEnvWithPatientLLMandRM',
    'MedicalConsultationEnvWithPatientLLMCategory',
]
