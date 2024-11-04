from flask import Flask, render_template, request, jsonify
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import os

# Set up environment variables for OpenAI API
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Use environment variable

# Database setup
engine = create_engine("sqlite:///social3.db")
db = SQLDatabase(engine=engine)
llm = ChatOpenAI(model="gpt-4o-mini")
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Flask app setup
app = Flask(__name__)

# Data dictionary for context
data_dictionary = """
| Column Name           | Description                                     |
|-----------------------|-------------------------------------------------|
| CASE_ID               | Unique identifier for each case                 |
| office_code           | Code representing the office location           |
| office_name           | Name of the office location                     |
| category_desc         | Description of the benefit category             |
| category_code         | Code representing the benefit category          |
| Town_No               | Town identification number                      |
| family_No             | Unique identifier for the family                |
| person_DOB            | Date of birth of the individual                 |
| age                   | Age of the individual                           |
| person_mobileNo       | Mobile number of the individual                 |
| Approval_Date         | Date when the benefit was approved              |
| person_gender         | Gender of the individual                        |
| GENDER_ARABIC         | Gender in Arabic                                |
| GENDER_ENGLISH        | Gender in English                               |
| Amount_of_help        | Amount of financial aid provided                |
| Percentage_of_help    | Percentage of total assistance allocated        |
| refund_amount         | Amount to be refunded                           |
| person_emirate_name   | Name of the individual's emirate                |
| person_emirate_code   | Code representing the individual's emirate      |
| area                  | Area of residence                               |
| PR_GROUP              | Group classification for the benefit program    |
| Education             | Educational qualification of the individual     |
| education_code        | Code representing education level               |
| marital_desc          | Marital status description                      |
| marital_code          | Code representing marital status                |
| LAST_PR               | Last benefit program received                   |
| Nationality           | Nationality of the individual                   |
| nationality_name      | Full name of the individual's nationality       |
| nationality_3_code    | 3-letter nationality code                       |
| nationality_2_code    | 2-letter nationality code                       |
| total_family_member   | Total number of family members                  |
| person_wifes          | Number of wives the individual has              |
| person_sons           | Number of sons                                  |
| person_daughters      | Number of daughters                             |
| person_no_relation    | Number of individuals with no direct relation   |
| person_sisters        | Number of sisters                               |
| person_brothers       | Number of brothers                              |
| INCOME_SOURCES        | Sources of income                               |
| PR_TYPE               | Type of benefit program                         |
| Case_Type             | Type of case                                    |
"""

@app.route('/')
def index():
    return render_template("index.html")  # HTML template for chat UI

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("user_input")
    input_text = f"Refer to the following data dictionary for context:\n\n{data_dictionary}\n\n{user_input}"
    result = agent_executor.invoke({"input": input_text})["output"]
    return jsonify({"response": result})

if __name__ == '__main__':
    app.run(debug=True)