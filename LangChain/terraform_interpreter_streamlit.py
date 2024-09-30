# Import necessary libraries
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Set up the Streamlit app
st.title("Terraform Plan Summary Generator")
st.write("Upload a Terraform plan output to generate a stakeholder-friendly summary of the planned infrastructure changes.\nHave a terraform project and generate a terraform output using the commands 'terraform plan -out=myPlan.plan' 'terraform show -no-color myPlan.plan > myPlan.txt'.")

# Input: Upload Terraform plan file
uploaded_file = st.file_uploader("Upload your Terraform plan file", type=["txt"])

if uploaded_file is not None:
  # Read Terraform plan from uploaded file
  terraform_plan_text = uploaded_file.read().decode('utf-8')

  # Display contents of the Terraform plan (optional)
  with st.expander("Show Terraform Plan"):
      st.text(terraform_plan_text)

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

  # Button to generate summary
  generate_summary_button = st.button("Generate Summary")
  if generate_summary_button:
      with st.spinner('Generating summary...'):
          # Invoke model and get result
          result = chain.invoke({"question": question})
          # Display the result
          st.header("Summary")
          st.write(result)

          # Create a text file with the summary
          summary_file = st.download_button(
              label="Download Summary",
              data=result,
              file_name="summary.txt",
              mime="text/plain"
          )

else:
  st.info("Please upload a Terraform plan file to get started.")
