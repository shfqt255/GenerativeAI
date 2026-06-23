from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage

#llm
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# code analyzer
@tool
def analyze_code(code: str) -> str:
    """
    Analyze code: explain what it does and its structure.
    """
    prompt = f"""
You are a senior software engineer.

Analyze this code:

{code}

Provide:
- Functionality explanation
- Key logic steps
- Readability comments
"""
    return llm.invoke(prompt).content


# bug detector
@tool
def detect_bugs(code: str) -> str:
    """
    Detect bugs and issues in code.
    """
    prompt = f"""
You are a debugging expert.

Find bugs or issues in this code:

{code}

Return:
- Bug description
- Why it is a problem
- Suggested fix
"""
    return llm.invoke(prompt).content


# improve suggestions
@tool
def improve_code(code: str) -> str:
    """
    Suggest optimizations and improvements.
    """
    prompt = f"""
You are a senior software architect.

Improve this code:

{code}

Return:
- Optimized version (if needed)
- Performance improvements
- Clean code suggestions
"""
    return llm.invoke(prompt).content


# tools bindings
tools = [analyze_code, detect_bugs, improve_code]

llm_with_tools = llm.bind_tools(tools)


#  query
query = """
Here is my code:

def add(a, b):
    return a + b

What do you think about it?
"""

# first llm call
response = llm_with_tools.invoke(query)

# tool execution
tool_map = {
    "analyze_code": analyze_code,
    "detect_bugs": detect_bugs,
    "improve_code": improve_code
}

tool_messages = []

if response.tool_calls:
    for call in response.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        tool_id = call["id"]

        result = tool_map[tool_name].invoke(tool_args)

        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_id
            )
        )

# final response
if tool_messages:
    final = llm_with_tools.invoke([response] + tool_messages)
else:
    final = response


# output
print(final.content)