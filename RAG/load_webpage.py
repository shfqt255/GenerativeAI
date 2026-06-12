from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv


url="https://www.fiverr.com/shfqt25?public_mode=true"
loader= WebBaseLoader(url)
docs= loader.load()

webpage_content = "\n\n".join(doc.page_content for doc in docs)

# print("-----docs---------")
# print(docs)

# print("------docs length---------")
# print(len(docs))

# print("---------loader---------")
# # Without calling .load(), printing the loader object displays its configuration details (URL, headers, etc.), not the loaded documents
# print(loader)


try:
    load_dotenv()
    model= ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    messages=[SystemMessage(content="""
You are a Fiverr profile analyst. I will give you a Fiverr profile URL. 
Analyze the profile and provide a structured response.

Please extract and present the following information in a clear, organized format:

## 1. BASIC INFORMATION
- **Name/Username**: 
- **Role/Title**: 
- **Location** (if available):
- **Languages** (if available):
- **Member Since**:
- **Response Time/Availability**:

## 2. WHAT HE SELLS
- **Primary Services/Gigs**: List each gig with:
  - Gig title
  - Starting price
  - Delivery time
  - Brief description

## 3. PORTFOLIO & PROJECT EXPERIENCE
- **Past Projects**: List notable projects with descriptions
- **Industries Served**: 
- **Notable Clients** (if mentioned):
- **Project Highlights**:

## 4. SKILLS & EXPERTISE
- **Technical Skills**:
- **Tools & Technologies**:
- **Certifications** (if any):

## 5. EDUCATION & QUALIFICATIONS
- **Education**:
- **Professional Certifications**:
- **Training/Courses**:

## 6. PERFORMANCE METRICS (if visible)
- **Rating**:
- **Number of Reviews**:
- **Completed Orders**:
- **Repeat Clients**:

## 7. PROFILE ANALYSIS & RECOMMENDATION

Based on the above information, provide a detailed analysis:

### Strengths:
- What does this freelancer excel at?
- What makes their profile stand out?

### Weaknesses:
- What's missing from their profile?
- Any red flags or concerns?

### Recommendation:
- ⭐ Rating out of 5: 
- **Hire / Don't Hire** recommendation
- **Best suited for**: (type of projects, client size, industry)
- **Not ideal for**: (where they fall short)
- **Confidence level** in this recommendation (High/Medium/Low)

### Tips for the Freelancer:
- 3-5 actionable improvements for their profile

Please be honest, specific, and provide reasoning for your recommendation.
"""),
HumanMessage(content=webpage_content),
]

    response=model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("\n\n----response----\n\n")
    print(response.content)
    
except Exception as exp:
    print(exp)