import os

from openai import OpenAI


class GPTAgent:
    """Thin wrapper around a real GPT‑4.1 model."""

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Did you put it in .env and pass --env-file .env?")

        self.client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""


def calculator_tool(expression: str) -> str:
    """Very small, unsafe calculator for simple expressions.

    Supports things like: 1+2, 3 * (4 + 5) / 2
    """

    try:
        # eval is fine here for learning in a controlled environment.
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {e}"


class Agent:
    """Single agent that can call tools or the LLM."""

    def __init__(self, llm: GPTAgent, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = {
            "calc": calculator_tool,
        }

    def handle(self, user_input: str) -> str:
        """Route the user input either to a tool or the LLM."""

        stripped = user_input.strip()
        lower = stripped.lower()

        # Tool call: "calc 1+2*3"
        for prefix, tool in self.tools.items():
            if lower.startswith(prefix + " "):
                expr = stripped[len(prefix) :].strip()
                return f"(tool={prefix}) {tool(expr)}"

        # Otherwise, ask the LLM
        return self.llm.generate(
            system_prompt=self.system_prompt,
            user_message=stripped,
        )


def main() -> None:
    system_prompt = (
        "You are a thoughtful assistant that reasons step by step. "
        "When the user asks general questions, answer clearly. "
        "When they give you results from the calculator tool, you may incorporate them into your reasoning."
    )
    llm = GPTAgent()
    agent = Agent(llm=llm, system_prompt=system_prompt)

    print("GPT‑4.1-powered agent with explicit tool routing.")
    print("Type 'calc <expression>' to use the calculator, or 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Goodbye.")
            break

        reply = agent.handle(user_input)
        print("\nAgent:")
        print(reply)
        print()


if __name__ == "__main__":
    main()

