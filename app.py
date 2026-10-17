"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from the project notebook. The verified query library, validation gate,
retry logic, execution and response-generation logic are carried over unchanged;
only the I/O (config loading, DB connection, user input and output display) is
adapted for Streamlit.

Run locally:   streamlit run app.py
"""

import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Northbridge Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("CREDIT_DB_PATH", APP_DIR / "credit_risk_portfolio.db"))
TEST_QUERIES_PATH = APP_DIR / "test_queries.csv"
AUDIT_LOG_PATH = APP_DIR / "audit_log.jsonl"


# ---------------------------------------------------------------------------
# Credentials and LLM setup
# (notebook: reads config.json and sets environment variables)
# Order of lookup: Streamlit secrets -> environment variables -> config.json
# ---------------------------------------------------------------------------
def load_credentials():
    api_key, api_base = None, None

    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass  # no secrets file configured

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

    config_file = APP_DIR / "config.json"
    if (not api_key) and config_file.exists():
        with open(config_file, "r") as file:
            config = json.load(file)
            api_key = config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")

    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()

if not OPENAI_API_KEY:
    st.error(
        "OpenAI credentials not found. Set OPENAI_API_KEY (and optionally OPENAI_API_BASE) "
        "in Streamlit secrets, as environment variables, or in a config.json file."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


@st.cache_resource(show_spinner=False)
def get_llms():
    llm_ = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_llm_ = ChatOpenAI(model="gpt-4o", temperature=0)
    return llm_, evaluator_llm_


llm, evaluator_llm = get_llms()


# ---------------------------------------------------------------------------
# Database connection (read-only)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_connection(db_path_str):
    uri = f"{Path(db_path_str).resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


if not DB_PATH.exists():
    st.error(f"Database file not found at: {DB_PATH}")
    st.stop()

conn = get_connection(str(DB_PATH))


# ---------------------------------------------------------------------------
# Database schema provided to the LLM in prompts
# ---------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ---------------------------------------------------------------------------
# Verified query template library
# ---------------------------------------------------------------------------
sql_1 = """
SELECT s.sector_name AS sector,
       ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_millions,
       ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard','Doubtful','Loss')
                      THEN l.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_exposure_millions
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_code, s.sector_name
ORDER BY SUM(l.total_outstanding) DESC;
"""

sql_2 = """
SELECT loan_category,
       ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions,
       COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY SUM(total_outstanding) DESC;
"""

sql_3 = """
SELECT ifrs9_stage,
       COUNT(*) AS loan_count,
       ROUND(SUM(ead_amount) / 1000000.0, 2) AS ead_millions,
       ROUND(SUM(ecl_amount) / 1000000.0, 2) AS ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage;
"""

sql_4 = """
SELECT s.sector_name AS sector,
       ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_coverage_ratio_pct
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_code, s.sector_name
ORDER BY AVG(p.provision_coverage_ratio) DESC;
"""

sql_5 = """
SELECT l.loan_account_number,
       l.borrower_name,
       s.sector_name AS sector,
       ROUND(l.total_outstanding / 1000000.0, 2) AS outstanding_millions,
       l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10;
"""

sql_6 = """
SELECT group_name,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_exposure_millions
FROM loan_master
WHERE group_name IS NOT NULL AND TRIM(group_name) <> ''
GROUP BY group_name
ORDER BY SUM(total_outstanding) DESC
LIMIT 5;
"""

sql_7 = """
SELECT l.loan_account_number,
       l.borrower_name,
       s.sector_name AS sector,
       ROUND(l.total_outstanding / 1000000.0, 2) AS outstanding_millions,
       l.days_past_due,
       l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC;
"""

sql_8 = """
SELECT CASE WHEN days_past_due = 0 THEN '0 (Current)'
            WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
            WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
            WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
            ELSE '90+' END AS dpd_bucket,
       COUNT(*) AS loan_count,
       ROUND(SUM(total_outstanding) / 1000000.0, 2) AS outstanding_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due);
"""

sql_9 = """
SELECT borrower_id,
       previous_rating,
       internal_rating AS current_rating,
       pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;
"""

sql_10 = """
SELECT reporting_date,
       ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date;
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },


    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },


    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },


    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },


    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },


    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },


    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },


    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },


    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },


    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ---------------------------------------------------------------------------
# Pipeline tools (logic carried over from the notebook)
# ---------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### ROLE
You are a query router for a supply chain operations analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording. For example, 'out of stock' means stockout (units_on_hand = 0), 'FTZ' or 'bonded' refers to is_cbp_bonded_ftz = 1, 'late' or 'behind schedule' means Delayed, 'facilities near capacity' means high occupancy.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer: a question asking for row-level detail (e.g., 'show me the shipments') should NOT match a template that returns an aggregate count.

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

### ROLE
You are a senior SQL developer specializing in supply chain operations analytics on a SQLite database.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Resolve named entities using warehouse_name or warehouse_id where relevant (for example, 'Dallas' maps to WH_DFW_01, 'Chicago' maps to WH_ORD_01).
5. Ensure the query is SQLite compatible.
6. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) — it silently returns 0; always use julianday(a)-julianday(b) for day differences.
7. Alias every numeric column with a suffix that states its unit, so the result is self-describing. Use _usd for dollar amounts, _pct or _percent for percentages, _count for counts, _hours for hour values, and _days for day values. Avoid bare aliases like "value", "amount", or "total".

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.

"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Normalise trim whitespace and a single trailing semicolon
    candidate_sql = candidate_sql.strip().rstrip(';').strip()

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result



    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]



    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result



    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )


    relevance_prompt = f"""
### ROLE
You are a senior data validator. Your job is to check whether a SQL query
correctly answers a business user's question about supply chain operations.

### CONTEXT
{track_context}

### INPUT
User Question: {user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result



    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            clean = lambda s: s.strip().rstrip(';').strip()
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({clean(expected_sql)}) LIMIT 0").description]
            actual_cols   = [d[0] for d in cur.execute(f"SELECT * FROM ({clean(candidate_sql)}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
### ROLE
You are a supply chain operations analyst writing a concise business response for a fulfillment or inventory question.

### INPUT
User Question: {user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific question directly. Do not dump the entire table.
2. If the user asked about a specific region, warehouse, carrier, or category, highlight only those rows.
3. Provide context from other rows only when it adds value (for example, ranking or comparison).
4. State exact numbers from the data. Do not round beyond what is shown.
5. Flag anything notable, such as a warehouse close to a capacity threshold or a carrier significantly worse than peers.
6. Use clear, professional language suitable for a fulfillment operations memo.
7. Keep the response focused. Two to four sentences for simple questions, up to a short paragraph for complex ones.
8. State the unit for every number, inferred from its column name: _usd as "$X", _pct or _percent as "X%", _count as a plain count, _hours as "X hours". Never state a bare number when the source column implies a unit.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.

{user_question}

{dataframe.to_string()}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, prints intermediate pipeline stages.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    if verbose:
        print(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        print(f"    Reason: {log['match_reason']}")


    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    if verbose:
        print(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")


    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    if verbose:
        print(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            print(f"    Failed check: {gate.get('failed_check')}")
            print(f"    Details: {gate.get('details')}")


    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            print(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        if verbose:
            print(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                print(f"    Retry failed check: {gate.get('failed_check')}")
                print(f"    Retry details: {gate.get('details')}")


    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        if verbose:
            print(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}


    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    if verbose:
        print(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            print(f"    Warnings: {exec_result['warnings']}")


    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative


    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    if verbose:
        print(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ---------------------------------------------------------------------------
# Streamlit helpers: audit trail, sample questions, rendering
# ---------------------------------------------------------------------------
def write_audit_record(log):
    """Append one audit record per pipeline run to a JSON-lines file."""
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **log}
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return True
    except OSError:
        return False  # e.g. read-only or ephemeral file system


def load_sample_questions():
    if TEST_QUERIES_PATH.exists():
        try:
            gt = pd.read_csv(TEST_QUERIES_PATH)
            return gt["User Query"].dropna().tolist()
        except Exception:
            pass
    return [
        "How much of our book is in real estate, and how much of that is non-performing?",
        "Show me how our overdue book is distributed across DPD buckets.",
        "List every restructured loan, flag which are impaired, and show the overall impaired percentage.",
        "Show me the average interest rate by sector, sorted from highest to lowest.",
        "How has the Stage 3 book size and its expected credit loss moved over the last four quarters?",
    ]


def format_confidence(conf):
    if isinstance(conf, (int, float)):
        return f"{conf:.2f}"
    if conf is None:
        return "n/a"
    return str(conf)


def render_trace(log):
    st.markdown(f"**[1] Intent classification:** route = `{log['route']}`, query_id = `{log['query_id']}`")
    if log.get("match_reason"):
        st.caption(log["match_reason"])

    built = "loaded from verified library" if log["route"] == "verified" else "generated fresh SQL"
    st.markdown(f"**[2] Query construction:** {built}")

    gate = log.get("gate_result") or {}
    st.markdown(
        f"**[3] Validation gate:** passed = `{gate.get('passed')}`, "
        f"relevance confidence = `{gate.get('relevance_confidence')}`"
    )
    if not gate.get("passed"):
        st.caption(f"Failed check: {gate.get('failed_check')} - {gate.get('details')}")
    else:
        st.caption(gate.get("details", ""))

    if log.get("retry_used"):
        st.markdown("**[4] Retry:** the generated SQL failed validation once and was regenerated.")
    if log.get("escalated"):
        st.markdown("**[!] Escalated to a human analyst.**")
    elif log.get("row_count") is not None:
        st.markdown(f"**[5] Execution:** {log['row_count']} rows returned")


def render_result(res):
    log = res["log"]
    df = res["dataframe"]

    if log["escalated"]:
        st.warning(
            "This question could not be reliably resolved by the engine and has been "
            "escalated to a human analyst."
        )
        st.write(log["narrative"])
    else:
        st.subheader("Answer")
        # Escape $ so amounts are not rendered as LaTeX by Streamlit markdown
        st.markdown(log["narrative"].replace("$", r"\$"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", str(log["route"]).capitalize())
    c2.metric("Query ID", log["query_id"] if log["query_id"] else "Generated")
    c3.metric("Confidence", format_confidence(log["confidence"]))
    c4.metric("Rows returned", log["row_count"] if log["row_count"] is not None else "-")

    tab_sql, tab_data, tab_trace, tab_audit = st.tabs(
        ["SQL used", "Raw data", "Validation & trace", "Audit record"]
    )

    with tab_sql:
        if log["escalated"]:
            st.caption("Rejected SQL (not executed):")
            st.code(log["candidate_sql"] or "-- no SQL produced", language="sql")
        else:
            st.code(log["executed_sql"], language="sql")

    with tab_data:
        if df is not None:
            st.dataframe(df)
            st.download_button(
                "Download result as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
            )
        else:
            st.info("No data returned (query escalated).")

    with tab_trace:
        render_trace(log)

    with tab_audit:
        st.json(json.loads(json.dumps(log, default=str)))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""
if "current" not in st.session_state:
    st.session_state["current"] = None
if "history" not in st.session_state:
    st.session_state["history"] = []


def set_question(q):
    st.session_state["question_input"] = q


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Sample questions")
    for i, q in enumerate(load_sample_questions()):
        st.button(q, key=f"sample_{i}", on_click=set_question, args=(q,))

    st.divider()
    st.header("System")
    st.caption(f"Database: `{DB_PATH.name}` (read-only)")
    st.caption("Router / generator: gpt-4o-mini")
    st.caption("Validator: gpt-4o")
    st.caption(f"Verified templates: {len(verified_query_library)}")

    with st.expander("Verified query library"):
        for qid, entry in verified_query_library.items():
            st.markdown(f"**{qid}** - {entry['description']}")

    st.divider()
    st.header("Session history")
    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df, hide_index=True)
        st.download_button(
            "Download session audit (CSV)",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="session_audit.csv",
            mime="text/csv",
        )
    else:
        st.caption("No questions asked yet.")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🏦 Northbridge Bank - Credit Risk Query Engine")
st.caption(
    "Ask routine commercial-lending portfolio questions in plain English. "
    "Every answer shows the SQL used, the raw data, and a confidence score, "
    "and every run is logged for audit. Queries are read-only."
)

question = st.text_area(
    "Your question",
    key="question_input",
    height=90,
    placeholder="e.g. How much of our book is in real estate, and how much of that is non-performing?",
)

run_clicked = st.button("Run query", type="primary")

if run_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running pipeline..."):
            try:
                result = run_pipeline(
                    question.strip(), conn, verified_query_library, database_schema, verbose=False
                )
                st.session_state["current"] = result
                logged = write_audit_record(result["log"])
                st.session_state["history"].append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "question": result["log"]["user_question"],
                    "route": result["log"]["route"],
                    "query_id": result["log"]["query_id"],
                    "confidence": format_confidence(result["log"]["confidence"]),
                    "rows": result["log"]["row_count"],
                    "escalated": result["log"]["escalated"],
                    "retry_used": result["log"]["retry_used"],
                })
                if not logged:
                    st.toast("Audit log file could not be written on this host; session history is still kept.")
            except Exception as e:
                st.session_state["current"] = None
                st.error(f"The pipeline hit an unexpected error and the question was not answered: {e}")

if st.session_state["current"] is not None:
    render_result(st.session_state["current"])
