import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

from langchain_tavily import TavilySearch



load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment")



tavily_tool = TavilySearch(
    max_results=10,
    topic="finance",
    search_depth="advanced",
    include_domains=[
        # Banking & Macro
        "reuters.com",
        "bloomberg.com",
        "ft.com",
        "federalreserve.gov",
        "bis.org",
        "rbi.org.in",
        "bankofengland.co.uk",

        # Financial Services
        "sebi.gov.in",
        "sec.gov",
        "morningstar.com",
        "investopedia.com",

        # Insurance
        "irdai.gov.in",
        "insurancejournal.com",
        "swissre.com",
        "munichre.com",

        # BFSI News
        "economictimes.indiatimes.com",
        "livemint.com",
        "businessinsider.com",
    ],
)


# --------------------------------------------------
# Prompt Templates
# --------------------------------------------------
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior BFSI industry analyst and financial journalist with 15+ years of experience.

Your expertise includes:

BANKING:
- Retail and corporate credit risk
- Digital banking and core modernization

FINANCIAL SERVICES:
- Asset management, mutual funds, capital markets
- SEBI regulations and ESG investing

INSURANCE:
- Life, health, general, reinsurance
- InsurTech (AI underwriting, embedded insurance)
- IRDAI regulations and actuarial modeling

MANDATORY:
- Always use the search tool before writing
- Validate data using 2–3 authoritative sources

STRUCTURE:
- Executive Summary (≤ 3 sentences)
- Industry Landscape
- Segment Deep Dive
- Case Studies
- Risks & Challenges
- 3–5 Year Outlook

Length: 900–1100 words
Always cite sources with year.
"""),
    ("user", "{topic}")
])


editor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior BFSI Content Editor.

Checklist:

1. COMPLIANCE
- No financial advice
- Add disclaimer if needed

2. ACCURACY
- Verify RBI, SEBI, IRDAI references
- Ensure numeric consistency

3. CLARITY
- Define acronyms (NPA, AUM, etc.)
- Simplify technical terms

4. STRUCTURE
- Improve flow
- Tighten executive summary
- Ensure smooth transitions

Do not change meaning.
"""),
    ("user", "{article}")
])


# --------------------------------------------------
# Agents
# --------------------------------------------------
writer_agent = create_agent(
    model="gpt-4.1-mini",
    tools=[tavily_tool],
)

editor_agent = create_agent(
    model="gpt-4.1-mini",
)



def run_bfsi_pipeline(topic: str) -> dict:

    print("\n" + "=" * 60)
    print(f"📊 BFSI Topic: {topic}")
    print("=" * 60 + "\n")

    # ---------------- Writer ----------------
    print("🔍 Writer Agent researching...\n")

    writer_messages = writer_prompt.format_messages(topic=topic)

    writer_result = writer_agent.invoke({
        "messages": writer_messages
    })

    written_content = writer_result["messages"][-1].content

    print("✅ Writer completed\n")

    # ---------------- Editor ----------------
    print("✏️ Editor refining...\n")

    editor_messages = editor_prompt.format_messages(article=written_content)

    editor_result = editor_agent.invoke({
        "messages": editor_messages
    })

    refined_content = editor_result["messages"][-1].content

    print("✅ Editor completed\n")

    return {
        "topic": topic,
        "draft": written_content,
        "final": refined_content
    }



BFSI_TOPICS = {
    1: "AI-Driven Credit Scoring in Indian Banks",
    2: "Impact of UPI on Banking Revenue",
    3: "Rise of Discount Brokers in India",
    4: "ESG Investing Risks and Opportunities",
    5: "InsurTech Disruption in India",
    6: "Parametric Insurance for Climate Risk",
    7: "Generative AI in BFSI",
    8: "Open Finance in India",
}


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️ Missing TAVILY_API_KEY")
        exit(1)

    print("\n📋 Topics:\n")
    for k, v in BFSI_TOPICS.items():
        print(f"{k}. {v}")

    choice = int(input("\nSelect topic (1–8) or 0 for custom: "))

    if choice == 0:
        topic = input("Enter custom topic: ")
    else:
        topic = BFSI_TOPICS.get(choice)

    result = run_bfsi_pipeline(topic)

    print("\n" + "=" * 60)
    print("📄 FINAL OUTPUT")
    print("=" * 60)

    print("\n📝 Draft:\n")
    print(result["draft"])

    print("\n✨ Final:\n")
    print(result["final"])