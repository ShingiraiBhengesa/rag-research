"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import json
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
    SearchAPI,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "model_provider", "max_tokens", "api_key", "base_url", "temperature", "seed", "reasoning_effort"),
)


def _model_config(configurable: Configuration, config: RunnableConfig, model_name: str, max_tokens: int):
    """Build LangChain model config, including optional local OpenAI-compatible endpoints."""
    model_config = {
        "model": model_name,
        "max_tokens": max_tokens,
        "api_key": configurable.model_api_key or get_api_key_for_model(model_name, config),
        "tags": ["langsmith:nostream"],
    }
    if configurable.model_provider:
        model_config["model_provider"] = configurable.model_provider
    if configurable.model_base_url:
        model_config["base_url"] = configurable.model_base_url
    if configurable.model_temperature is not None:
        model_config["temperature"] = configurable.model_temperature
    if configurable.model_seed is not None:
        model_config["seed"] = configurable.model_seed
    if configurable.model_reasoning_effort is not None:
        model_config["reasoning_effort"] = configurable.model_reasoning_effort
    return model_config


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences

    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = _model_config(
        configurable, config, configurable.research_model, configurable.research_model_max_tokens
    )
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to research supervisor with initialized context
    """
    configurable = Configuration.from_runnable_config(config)
    replay_brief = _v3_replay_research_brief(configurable)
    if _is_benchmark_local_search(configurable) and replay_brief:
        research_brief = replay_brief
    else:
        # Step 1: Set up the research model for structured output
        research_model_config = _model_config(
            configurable, config, configurable.research_model, configurable.research_model_max_tokens
        )

        # Configure model for structured research question generation
        research_model = (
            configurable_model
            .with_structured_output(ResearchQuestion)
            .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
            .with_config(research_model_config)
        )

        # Step 2: Generate structured research brief from user messages
        prompt_content = transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        )
        response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
        research_brief = response.research_brief
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    replay_topic = _v3_replay_research_topic(configurable)
    if (
        _is_benchmark_local_search(configurable)
        and replay_topic
        and not _has_conduct_research_observation(supervisor_messages)
    ):
        tool_call = {
            "name": "ConductResearch",
            "args": {"research_topic": replay_topic},
            "id": _v3_replay_supervisor_tool_call_id(configurable, state.get("research_iterations", 0)),
        }
        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [AIMessage(content="", tool_calls=[tool_call])],
                "research_iterations": state.get("research_iterations", 0) + 1,
            },
        )

    research_model_config = _model_config(
        configurable, config, configurable.research_model, configurable.research_model_max_tokens
    )
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    response = _sanitize_ai_message_for_vllm_history(
        await research_model.ainvoke(
            _messages_for_model(configurable, configurable.research_model, supervisor_messages)
        ),
        configurable,
        configurable.research_model,
    )
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    tool_calls = [
        tool_call
        for tool_call in (getattr(most_recent_message, "tool_calls", None) or [])
        if isinstance(tool_call, dict)
    ]
    valid_conduct_research_calls = [
        tool_call
        for tool_call in tool_calls
        if _tool_call_name(tool_call) == "ConductResearch"
        and _tool_call_arg(tool_call, "research_topic")
    ]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not tool_calls
    conduct_research_attempt = any(
        _tool_call_name(tool_call) == "ConductResearch"
        for tool_call in tool_calls
    )
    conduct_research_tool_call = bool(valid_conduct_research_calls)
    research_complete_tool_call = any(
        _tool_call_name(tool_call) == "ResearchComplete"
        for tool_call in tool_calls
    )

    benchmark_local_requires_initial_research = (
        _is_benchmark_local_search(configurable)
        and not exceeded_allowed_iterations
        and (
            no_tool_calls
            or research_complete_tool_call
            or (conduct_research_attempt and not conduct_research_tool_call)
            or (research_iterations > 1 and not conduct_research_tool_call)
        )
        and not _has_conduct_research_observation(supervisor_messages)
    )
    if benchmark_local_requires_initial_research:
        return await _forced_initial_benchmark_research(state, config, research_iterations)

    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in tool_calls
        if _tool_call_name(tool_call) == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        args = _tool_call_args(tool_call) or {}
        reflection_content = str(args.get("reflection") or "No reflection content supplied.").strip()
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=_tool_call_id(tool_call)
        ))
    
    # Handle ConductResearch calls (research delegation)
    invalid_conduct_research_calls = [
        tool_call
        for tool_call in tool_calls
        if _tool_call_name(tool_call) == "ConductResearch"
        and not _tool_call_arg(tool_call, "research_topic")
    ]
    for tool_call in invalid_conduct_research_calls:
        all_tool_messages.append(ToolMessage(
            content="Error: ConductResearch requires a non-empty research_topic argument.",
            name="ConductResearch",
            tool_call_id=_tool_call_id(tool_call),
        ))
    conduct_research_calls = valid_conduct_research_calls
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # Execute research tasks in parallel
            research_tasks = []
            for delegation_index, tool_call in enumerate(allowed_conduct_research_calls):
                args = _tool_call_args(tool_call) or {}
                research_topic = str(args.get("research_topic") or "").strip()
                tool_call_id = _tool_call_id(tool_call, f"conduct_research_{delegation_index}")
                researcher_subgraph_id = (
                    f"supervisor_iter_{research_iterations}:"
                    f"delegation_{delegation_index}:"
                    f"tool_call_{tool_call_id}"
                )
                research_tasks.append(
                    researcher_subgraph.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=research_topic)
                        ],
                        "research_topic": research_topic,
                        "v3_supervisor_delegation_index": delegation_index,
                        "v3_supervisor_tool_call_id": tool_call_id,
                        "v3_researcher_subgraph_id": researcher_subgraph_id,
                    }, config)
                )
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=_tool_call_name(tool_call),
                    tool_call_id=_tool_call_id(tool_call)
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=_tool_call_id(overflow_call)
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.research_model):
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
            raise
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    replay_queries = _v3_replay_first_search_queries(configurable)
    if (
        _is_benchmark_local_search(configurable)
        and replay_queries
        and not _has_benchmark_local_search_observation(researcher_messages)
    ):
        tool_name, _ = _benchmark_local_search_tool(tools)
        tool_call = {
            "name": tool_name,
            "args": {"queries": replay_queries},
            "id": _v3_replay_search_tool_call_id(configurable, state.get("tool_call_iterations", 0)),
        }
        return Command(
            goto="researcher_tools",
            update={
                "researcher_messages": [AIMessage(content="", tool_calls=[tool_call])],
                "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
            },
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = _model_config(
        configurable, config, configurable.research_model, configurable.research_model_max_tokens
    )
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    try:
        response = _sanitize_ai_message_for_vllm_history(
            await research_model.ainvoke(
                _messages_for_model(configurable, configurable.research_model, messages)
            ),
            configurable,
            configurable.research_model,
        )
    except Exception as e:
        if (
            _is_benchmark_local_search(configurable)
            and _has_benchmark_local_search_observation(researcher_messages)
            and is_token_limit_exceeded(e, configurable.research_model)
        ):
            return Command(goto="compress_research")
        raise
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


BENCHMARK_LOCAL_SEARCH_TOOL_NAMES = {"tavily_search", "benchmark_search", "web_search"}


def _is_benchmark_local_search(configurable: Configuration) -> bool:
    search_api = configurable.search_api
    return getattr(search_api, "value", search_api) == SearchAPI.BENCHMARK_LOCAL.value


def _message_name(message) -> str:
    if isinstance(message, dict):
        return str(message.get("name") or "")
    return str(getattr(message, "name", "") or "")


def _message_content(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _has_benchmark_local_search_observation(messages) -> bool:
    for message in messages:
        if _message_name(message) in BENCHMARK_LOCAL_SEARCH_TOOL_NAMES and "benchmark_local" in _message_content(message):
            return True
    return False


def _has_conduct_research_observation(messages) -> bool:
    return any(_message_name(message) == "ConductResearch" for message in messages)


def _v3_replay_research_brief(configurable: Configuration) -> str:
    return str(configurable.v3_replay_research_brief or "").strip()


def _v3_replay_research_topic(configurable: Configuration) -> str:
    return str(configurable.v3_replay_research_topic or "").strip()


def _v3_replay_first_search_queries(configurable: Configuration) -> list[str]:
    return [str(query) for query in configurable.v3_replay_first_search_queries or [] if str(query).strip()]


def _v3_replay_supervisor_tool_call_id(configurable: Configuration, iteration: int) -> str:
    return str(configurable.v3_replay_supervisor_tool_call_id or f"v3_replay_conduct_research_{iteration}")


def _v3_replay_search_tool_call_id(configurable: Configuration, iteration: int) -> str:
    return str(configurable.v3_replay_search_tool_call_id or f"v3_replay_benchmark_search_{iteration}")


def _coerce_tool_call_args(args):
    if isinstance(args, dict):
        return args
    if args in (None, ""):
        return {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args, strict=False)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else {}
    return None


def _tool_call_name(tool_call) -> str:
    if not isinstance(tool_call, dict):
        return ""
    return str(tool_call.get("name") or "")


def _tool_call_id(tool_call, fallback: str = "tool_call") -> str:
    if not isinstance(tool_call, dict):
        return fallback
    return str(tool_call.get("id") or fallback)


def _tool_call_args(tool_call):
    if not isinstance(tool_call, dict):
        return None
    return _coerce_tool_call_args(tool_call.get("args"))


def _tool_call_arg(tool_call, key: str) -> str:
    args = _tool_call_args(tool_call)
    if not isinstance(args, dict):
        return ""
    return str(args.get(key) or "").strip()


def _plain_tool_call_summary(tool_call, detail_limit: int = 300) -> str:
    """Renders a past tool call as prose for gemma's plain-alternating history,
    never as a machine-parseable template.

    2026-09-28 audit of gemma27b production traces found the model literally
    copying the previous rendering here -- a JSON dict of the exact shape
    {"name": ..., "args": ..., "id": ...} -- back out as its own response
    text on a LATER turn, instead of emitting a new, real tool call or a
    final answer. Because the copied text is plain content (its own
    tool_calls= is empty), it was silently never executed, yet the episode
    looked to a naive read like the researcher had taken an action. 93% of a
    60-row sample of gemma27b's "no further action" rows were this exact
    reconstructable pattern. Removing the reproducible JSON template removes
    the thing being imitated; this conveys the same information (what tool,
    what it was asked) in prose that also explicitly tells the model this
    already happened and must not be repeated as text.
    """
    name = _tool_call_name(tool_call) or "unknown"
    args = _tool_call_args(tool_call)
    detail = ""
    if isinstance(args, dict) and args:
        if name == "tavily_search" and isinstance(args.get("queries"), list) and args["queries"]:
            detail = "; searched for: " + "; ".join(str(q) for q in args["queries"])
        elif name == "think_tool" and args.get("reflection"):
            detail = "; reflected: " + str(args["reflection"])
        else:
            detail = "; arguments were: " + ", ".join(f"{k}={v!r}" for k, v in args.items())
    if len(detail) > detail_limit:
        detail = detail[:detail_limit].rstrip() + "..."
    return (
        f"(Note: the assistant already used the {name} tool once{detail}. "
        "This already happened -- its result, if any, appears elsewhere in "
        "this history. Do not write text describing or repeating this tool "
        "call. Either make a new, different tool call now, or write the "
        "final answer.)"
    )


def _benchmark_local_fallback_research_summary(raw_notes_content: str) -> str:
    evidence = str(raw_notes_content or "").strip()
    if len(evidence) > 8000:
        evidence = evidence[:8000].rstrip() + "\n\n[raw benchmark-local notes truncated after token-limit fallback]"
    return (
        "Benchmark-local retrieval evidence preserved after compression token-limit fallback."
        f"\n\n{evidence}"
    )


def _benchmark_local_final_report_instruction(attempt: int) -> str:
    instruction = (
        "\n\nOutput requirement: Begin the visible assistant message immediately "
        "with `Answer: ` followed by a concise direct answer to the research brief. "
        "Put this answer before any title or heading. If the findings are insufficient, "
        "state that explicitly on the Answer: line. Then write the report."
    )
    if attempt <= 0:
        return instruction
    return (
        instruction
        + "\n\nThe previous report-generation attempt returned an empty visible "
        "message. Do not leave the assistant content blank. Do not answer only in "
        "hidden reasoning. The visible response must start with `Answer: `."
    )


def _truncate_for_empty_final_report_retry(findings: str, attempt: int) -> str:
    text = str(findings or "")
    if not text:
        return text
    # GPT-OSS can occasionally emit hidden reasoning with no visible final text
    # on long writer prompts. Keep the prompt progressively smaller on retries.
    limits = (20000, 12000, 8000)
    limit = limits[min(max(attempt - 1, 0), len(limits) - 1)]
    if len(text) <= limit:
        return text
    head = text[: limit // 2].rstrip()
    tail = text[-limit // 2 :].lstrip()
    return head + "\n\n[findings truncated for empty final-report retry]\n\n" + tail


def _messages_for_model(configurable: Configuration, model_name: str, messages):
    """Adapt message history only for model templates that reject tool roles."""
    if _requires_plain_alternating_history(configurable, model_name):
        return _plain_alternating_messages(messages)
    return messages


def _requires_plain_alternating_history(configurable: Configuration, model_name: str) -> bool:
    model_text = str(model_name or "").lower()
    return bool(configurable.model_base_url) and "gemma" in model_text


def _requires_single_tool_call_history(configurable: Configuration | None, model_name: str | None) -> bool:
    model_text = str(model_name or "").lower()
    return bool(configurable and configurable.model_base_url) and "llama" in model_text


def _plain_alternating_messages(messages):
    system_parts = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system_parts.append(_message_content(message).strip())
            continue
        role, content = _plain_turn_from_message(message)
        _append_plain_turn(turns, role, content)

    if turns and turns[0][0] != "user":
        turns.insert(0, ("user", "Continue from the conversation context."))
    if turns and turns[-1][0] != "user":
        _append_plain_turn(turns, "user", "Continue.")

    adapted = []
    system_content = "\n\n".join(part for part in system_parts if part)
    if system_content:
        adapted.append(SystemMessage(content=system_content))
    for role, content in turns:
        adapted.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))
    return adapted


def _plain_turn_from_message(message) -> tuple[str, str]:
    if isinstance(message, ToolMessage):
        name = _message_name(message) or "tool"
        tool_call_id = str(getattr(message, "tool_call_id", "") or "")
        label = f"Tool result from {name}"
        if tool_call_id:
            label += f" ({tool_call_id})"
        return "user", f"{label}:\n{_message_content(message)}"

    if isinstance(message, AIMessage):
        content_parts = []
        content = _message_content(message).strip()
        if content:
            content_parts.append(content)
        for tool_call in message.tool_calls or []:
            content_parts.append(_plain_tool_call_summary(tool_call))
        return "assistant", "\n\n".join(content_parts) or "Assistant made no textual response."

    return "user", _message_content(message)


def _append_plain_turn(turns: list[tuple[str, str]], role: str, content: str) -> None:
    content = str(content or "").strip()
    if not content:
        return
    if turns and turns[-1][0] == role:
        turns[-1] = (role, turns[-1][1].rstrip() + "\n\n" + content)
    else:
        turns.append((role, content))


def _sanitize_ai_message_for_vllm_history(
    message,
    configurable: Configuration | None = None,
    model_name: str | None = None,
):
    if not isinstance(message, AIMessage):
        return message

    changed = False
    seen_ids = set()
    sanitized_tool_calls = []
    for tool_call in message.tool_calls or []:
        tool_call_id = tool_call.get("id")
        args = _coerce_tool_call_args(tool_call.get("args"))
        if args is None or not tool_call.get("name") or not tool_call_id:
            changed = True
            continue
        sanitized_call = dict(tool_call)
        sanitized_call["args"] = args
        sanitized_tool_calls.append(sanitized_call)
        seen_ids.add(tool_call_id)
        changed = changed or args != tool_call.get("args")

    for invalid_call in getattr(message, "invalid_tool_calls", None) or []:
        tool_call_id = invalid_call.get("id")
        if tool_call_id in seen_ids:
            changed = True
            continue
        args = _coerce_tool_call_args(invalid_call.get("args"))
        if args is None or not invalid_call.get("name") or not tool_call_id:
            changed = True
            continue
        sanitized_tool_calls.append(
            {
                "name": invalid_call["name"],
                "args": args,
                "id": tool_call_id,
                "type": "tool_call",
            }
        )
        seen_ids.add(tool_call_id)
        changed = True

    additional_kwargs = dict(message.additional_kwargs or {})
    if "tool_calls" in additional_kwargs or "function_call" in additional_kwargs:
        additional_kwargs.pop("tool_calls", None)
        additional_kwargs.pop("function_call", None)
        changed = True

    if _requires_single_tool_call_history(configurable, model_name) and len(sanitized_tool_calls) > 1:
        sanitized_tool_calls = sanitized_tool_calls[:1]
        changed = True

    if not changed:
        return message
    return message.model_copy(
        update={
            "tool_calls": sanitized_tool_calls,
            "invalid_tool_calls": [],
            "additional_kwargs": additional_kwargs,
        }
    )


def _tool_name(tool) -> str:
    return str(tool.name if hasattr(tool, "name") else tool.get("name", "web_search"))


def _benchmark_local_search_tool(tools):
    for candidate in tools:
        metadata = getattr(candidate, "metadata", None) or {}
        if isinstance(metadata, dict) and metadata.get("backend") == "benchmark_local":
            return _tool_name(candidate), candidate
    for candidate in tools:
        name = _tool_name(candidate)
        if name in BENCHMARK_LOCAL_SEARCH_TOOL_NAMES:
            return name, candidate
    raise ValueError("benchmark_local search is configured but no benchmark-local search tool is available")


def _benchmark_local_query(state: ResearcherState, researcher_messages) -> str:
    query = str(state.get("research_topic") or "").strip()
    if not query and researcher_messages:
        query = _message_content(researcher_messages[-1]).strip()
    return " ".join(query.split())[:1000] or "benchmark question"


async def _forced_benchmark_local_search_messages(
    state: ResearcherState,
    config: RunnableConfig,
    tools,
) -> list:
    configurable = Configuration.from_runnable_config(config)
    tool_name, search_tool = _benchmark_local_search_tool(tools)
    iteration = state.get("tool_call_iterations", 0)
    replay_queries = _v3_replay_first_search_queries(configurable)
    search_queries = (
        replay_queries
        if replay_queries and not _has_benchmark_local_search_observation(state.get("researcher_messages", []))
        else [_benchmark_local_query(state, state.get("researcher_messages", []))]
    )
    tool_call = {
        "name": tool_name,
        "args": {"queries": search_queries},
        "id": (
            _v3_replay_search_tool_call_id(configurable, iteration)
            if replay_queries
            else f"forced_benchmark_local_search_{iteration}"
        ),
    }
    observation = await execute_tool_safely(
        search_tool,
        tool_call["args"],
        _config_with_v3_tool_context(config, state, tool_call, 0),
    )
    return [
        AIMessage(content="", tool_calls=[tool_call]),
        ToolMessage(content=observation, name=tool_name, tool_call_id=tool_call["id"]),
    ]


async def _forced_initial_benchmark_research(
    state: SupervisorState,
    config: RunnableConfig,
    research_iterations: int,
) -> Command[Literal["supervisor"]]:
    research_topic = " ".join(str(state.get("research_brief") or "").split()) or "Research the benchmark-local question."
    tool_call = {
        "name": "ConductResearch",
        "args": {"research_topic": research_topic},
        "id": f"forced_conduct_research_{research_iterations}",
    }
    researcher_subgraph_id = (
        f"supervisor_iter_{research_iterations}:"
        "delegation_0:"
        f"tool_call_{tool_call['id']}"
    )
    observation = await researcher_subgraph.ainvoke(
        {
            "researcher_messages": [HumanMessage(content=research_topic)],
            "research_topic": research_topic,
            "v3_supervisor_delegation_index": 0,
            "v3_supervisor_tool_call_id": tool_call["id"],
            "v3_researcher_subgraph_id": researcher_subgraph_id,
        },
        config,
    )
    tool_messages = [
        AIMessage(content="", tool_calls=[tool_call]),
        ToolMessage(
            content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
            name="ConductResearch",
            tool_call_id=tool_call["id"],
        ),
    ]
    update_payload = {"supervisor_messages": tool_messages}
    raw_notes_concat = "\n".join(observation.get("raw_notes", []))
    if raw_notes_concat:
        update_payload["raw_notes"] = [raw_notes_concat]
    return Command(goto="supervisor", update=update_payload)


def _tool_calls_include_search(tool_calls) -> bool:
    return any(_tool_call_name(tool_call) in BENCHMARK_LOCAL_SEARCH_TOOL_NAMES for tool_call in tool_calls)


def _config_with_v3_tool_context(config: RunnableConfig, state: ResearcherState, tool_call: dict, tool_call_index: int):
    """Attach passive trace metadata for v3 local-retrieval experiments."""
    updated_config = dict(config or {})
    configurable = dict(updated_config.get("configurable", {}))
    configurable["v3_tool_context"] = {
        "supervisor_delegation_index": state.get("v3_supervisor_delegation_index"),
        "supervisor_tool_call_id": state.get("v3_supervisor_tool_call_id"),
        "researcher_subgraph_id": state.get("v3_researcher_subgraph_id"),
        "tool_call_iteration": state.get("tool_call_iterations", 0),
        "tool_call_index": tool_call_index,
        "tool_call_id": tool_call.get("id"),
        "tool_name": tool_call.get("name"),
    }
    updated_config["configurable"] = configurable
    return updated_config


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    force_benchmark_search = (
        _is_benchmark_local_search(configurable)
        and not _has_benchmark_local_search_observation(researcher_messages)
    )
    
    if not has_tool_calls and not has_native_search:
        if force_benchmark_search:
            tools = await get_all_tools(config)
            return Command(
                goto="researcher",
                update={"researcher_messages": await _forced_benchmark_local_search_messages(state, config, tools)},
            )
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    raw_tool_calls = [
        tool_call
        for tool_call in (getattr(most_recent_message, "tool_calls", None) or [])
        if isinstance(tool_call, dict)
    ]
    tool_calls_with_args = []
    invalid_tool_outputs = []
    for tool_call_index, tool_call in enumerate(raw_tool_calls):
        tool_name = _tool_call_name(tool_call)
        tool_args = _tool_call_args(tool_call)
        tool_call_id = _tool_call_id(tool_call, f"tool_call_{tool_call_index}")
        if not tool_name or tool_args is None:
            invalid_tool_outputs.append(ToolMessage(
                content="Error executing tool: malformed tool call arguments.",
                name=tool_name or "unknown_tool",
                tool_call_id=tool_call_id,
            ))
            continue
        if tool_name not in tools_by_name:
            invalid_tool_outputs.append(ToolMessage(
                content=f"Error executing tool: unknown tool {tool_name!r}.",
                name=tool_name,
                tool_call_id=tool_call_id,
            ))
            continue
        tool_calls_with_args.append((tool_call, tool_args))

    tool_calls = [tool_call for tool_call, _ in tool_calls_with_args]
    tool_execution_tasks = [
        execute_tool_safely(
            tools_by_name[_tool_call_name(tool_call)],
            tool_args,
            _config_with_v3_tool_context(config, state, tool_call, tool_call_index),
        )
        for tool_call_index, (tool_call, tool_args) in enumerate(tool_calls_with_args)
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = invalid_tool_outputs + [
        ToolMessage(
            content=observation,
            name=_tool_call_name(tool_call),
            tool_call_id=_tool_call_id(tool_call)
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    if force_benchmark_search and not _tool_calls_include_search(tool_calls):
        tool_outputs.extend(await _forced_benchmark_local_search_messages(state, config, tools))
        return Command(
            goto="researcher",
            update={"researcher_messages": tool_outputs}
        )
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        _tool_call_name(tool_call) == "ResearchComplete"
        for tool_call in raw_tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config(
        _model_config(configurable, config, configurable.compression_model, configurable.compression_model_max_tokens)
    )
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(
                _messages_for_model(configurable, configurable.compression_model, messages)
            )
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    compressed_research = (
        _benchmark_local_fallback_research_summary(raw_notes_content)
        if _is_benchmark_local_search(configurable) and raw_notes_content.strip()
        else "Error synthesizing research report: Maximum retries exceeded"
    )
    
    return {
        "compressed_research": compressed_research,
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = _model_config(
        configurable, config, configurable.final_report_model, configurable.final_report_model_max_tokens
    )
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            report_writer = configurable_model.with_config(writer_model_config)
            if _is_benchmark_local_search(configurable):
                final_report_prompt += _benchmark_local_final_report_instruction(current_retry)
                # Observed with gpt-oss-20b (benchmark-local production gate,
                # 2026-09-08): this call binds no tools, yet the model
                # sometimes still emits a structured tool_calls response
                # (e.g. a hallucinated ConductResearch call, a supervisor-
                # level tool name that appeared earlier in its context) with
                # empty `content`, which the empty-response retry above
                # cannot distinguish from a genuinely empty generation.
                # tool_choice="none" tells the server to never parse tool
                # calls for this request regardless of what the model's raw
                # output looks like, forcing the response into `content`
                # where the retry/truncation logic can actually see it.
                report_writer = report_writer.bind(tool_choice="none")

            # Generate the final report
            final_report = await report_writer.ainvoke([
                HumanMessage(content=final_report_prompt)
            ])

            final_report_text = _message_content(final_report).strip()
            if not final_report_text:
                current_retry += 1
                if current_retry <= max_retries:
                    findings = _truncate_for_empty_final_report_retry(findings, current_retry)
                    continue
                return {
                    "final_report": "Error generating final report: empty model response",
                    "messages": [AIMessage(content="Report generation failed due to an empty model response")],
                    **cleared_state
                }
            
            # Return successful report generation
            return {
                "final_report": final_report_text,
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
