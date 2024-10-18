from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger


def create_speaker_agent(llm, agent_name):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are {agent_name}, an AI marketing expert. Your task is to propose innovative marketing strategies based on the given product information.
        Consider the following aspects in your strategy:
        1. Target audience
        2. Unique selling propositions
        3. Marketing channels (e.g., social media, email, content marketing)
        4. Brand positioning
        5. Key messaging
        6. Call to action
        Provide a comprehensive strategy that addresses these points."""),
        HumanMessage(content="""Create a marketing strategy for this product:
        Product Name: {product_name}
        Product Type: {product_type}
        Key Features: {features}
        Target Audience: {target_audience}
        Unique Selling Proposition: {usp}

        Please provide a detailed marketing strategy based on this information.""")
    ])

    def speaker_agent(product_info):
        logger.debug(f"{agent_name} received product info: {product_info}")
        result = llm.invoke(prompt.format(**product_info))
        logger.debug(f"{agent_name} strategy: {result.content}")
        return result

    return speaker_agent


def create_judge_agent(llm):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a judge evaluating marketing strategies. Your task is to analyze the proposals and declare a winner based on the following criteria:
        1. Alignment with the product and target audience
        2. Creativity and innovation
        3. Potential effectiveness and reach
        4. Clarity and coherence of the strategy
        5. Utilization of the product's unique selling propositions
        Provide a detailed evaluation and justification for your decision."""),
        HumanMessage(content="""For the following product:
        Product Name: {product_name}
        Product Type: {product_type}
        Key Features: {features}
        Target Audience: {target_audience}
        Unique Selling Proposition: {usp}

        Evaluate these marketing strategies and declare a winner:

        Strategy 1: {strategy1}

        Strategy 2: {strategy2}

        Please provide a detailed evaluation and declare a winner with justification.""")
    ])

    def judge_agent(product_info, strategy1, strategy2):
        logger.debug(f"Judge received product info: {product_info}")
        logger.debug(f"Judge received strategy 1: {strategy1}")
        logger.debug(f"Judge received strategy 2: {strategy2}")
        result = llm.invoke(prompt.format(**product_info, strategy1=strategy1, strategy2=strategy2))
        logger.debug(f"Judge decision: {result.content}")
        return result

    return judge_agent