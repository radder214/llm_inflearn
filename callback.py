from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List

class LLMDebugCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # 호출 위치 추적을 위한 스택 정보 가져오기
        import traceback
        stack = traceback.extract_stack()
        # 콜백 관련 프레임을 제외한 직전 호출 위치 찾기
        caller_frame = stack[-3]  # 필요에 따라 인덱스 조정 가능
        
        print("\n" + "="*50)
        print(f"🔍 LLM 호출 위치: {caller_frame.filename}:{caller_frame.lineno}")
        print(f"📝 프롬프트:")
        for prompt in prompts:
            print(prompt)
        print("="*50 + "\n")