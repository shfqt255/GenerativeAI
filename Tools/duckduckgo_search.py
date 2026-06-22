from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


# load apis
load_dotenv()

# llm
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash'
)

# search_tool
search_tool= DuckDuckGoSearchResults()

#query
query = "Find latest news regarding Pakistan politics"

# invoke search_tool
search_results=search_tool.invoke(query)

# write prompt
prompt= PromptTemplate.from_template(
    """

Role: You are a neutral, objective political analyst specializing in Pakistan and South Asian affairs, use only the search results.

Task: Using the search results provided below, generate a concise, fact-based briefing on the latest developments in Pakistan's politics.

Input Data:
`{search_results}`

Instructions:
- Synthesize, don't copy-paste. Read all the search results and extract the most important, actionable, and verified information.
- Structure your output as bullet points organized under clear thematic headings (e.g., "Imran Khan & PTI Legal Battles," "Economic Policy & IMF," "Military-Civilian Relations," "Provincial Politics," "Foreign Relations").
- Chronological order: Within each theme, order bullet points from newest to oldest where possible.
- Attribution: Briefly mention the source for each major claim (e.g., "According to Dawn," "As per Geo News reports").
- Verification: If the search results contain conflicting reports, mention both sides clearly (e.g., "While source A states X, source B reports Y").
- Tone: Strictly neutral, professional, and free of editorial commentary or speculation.

Output Format:
- Start with a 1-sentence summary of the most dominant political story right now.
- Follow with thematic bullet points (use bold text for the theme title).
- End with 2-3 "Key Watchpoints" — questions or trends to monitor in the coming days.

Constraints:
- Do not use fluff or filler sentences.
- Do not express your own opinions or predictions.
- If the search results are outdated (older than 3 days), state that clearly at the top.
- If the results are insufficient to cover a topic, write "No recent information available on [topic]."

---

Example of the expected output structure:

---

Summary: The current political landscape is dominated by the Supreme Court's hearing on the PTI's petition regarding election results, while the government focuses on IMF negotiations.

1. Supreme Court & Election Disputes
- The Supreme Court adjourned the hearing on the PTI's petition challenging the Punjab election results until Thursday, citing procedural issues. *(Dawn)*
- Justice XYZ remarked that the election commission's actions were "questionable," prompting a response from the ECP. *(Geo News)*

2. Economic Policy & IMF
- The Finance Minister confirmed that the IMF mission has completed its review, with the next tranche expected by month-end. *(The Express Tribune)*
- Fuel prices were hiked by Rs 10 per liter, aligning with IMF conditions, sparking protests in Karachi. *(ARY News)*

3. Military-Civilian Relations
- Army Chief General Asim Munir met with the Prime Minister to discuss the security situation in the Balochistan border region. *(Dawn)*

Key Watchpoints:
- Will the Supreme Court set a new date for the election challenge this week?
- How will the public react to the fuel price hikes over the weekend?
- Is there any official update on the repatriation of illegal immigrants?"""
)

chain= prompt| llm| StrOutputParser()

result = chain.invoke({"search_results": search_results})
print(result)


