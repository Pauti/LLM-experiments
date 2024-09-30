from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Read Terraform plan
with open('/Users/paulstrebenitzer/Projects/AI/LangChain/terraform_plan.txt', 'r') as file:
    terraform_plan_text = file.read()

# Define prompt template
template = """
Question: {question}

The following is a Terraform plan output, detailing the upcoming changes to the infrastructure:
+ create new resources
~ update existing resources in place
- destroy existing resources
-/+ destroy and then create a replacement resource

Please provide a clear and accurate summary of the changes that Terraform will implement. Your summary should be suitable for stakeholders, highlighting the key actions and their impact on the infrastructure. Be precise and avoid technical jargon where possible, ensuring the explanation is understandable for non-technical audiences.

Answer: Provide a stakeholder-friendly summary of the planned infrastructure changes, focusing on what will change, why it matters, and any potential impacts.
"""

# Formulate question
question = f"Based on the following Terraform plan, what changes will be performed, and what should stakeholders know about these changes?\n\n{terraform_plan_text}"

# Create prompt and model
prompt = ChatPromptTemplate.from_template(template)
model = Ollama(model="llama3:latest")

# Chain prompt and model
chain = prompt | model

# Invoke model and get result
result = chain.invoke({"question": question})

print(result)