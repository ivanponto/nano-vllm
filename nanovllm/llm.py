from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    LLM 类是整个推理框架的高层入口。
    它继承自 LLMEngine，提供了与 vLLM 风格类似的 API 接口。
    用户可以通过这个类直接加载模型并进行文本生成。
    """
    pass
