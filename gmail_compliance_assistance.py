
!pip install -q transformers sentence-transformers
!pip install -q langchain langchain-community langchain-huggingface
!pip install -q faiss-gpu   # or faiss-cpu if no GPU
!pip install --upgrade google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
!pip install -q torch accelerate bitsandbytes

from huggingface_hub import login

#  Replace with your own HF token
login("HF_Token")

from google.colab import drive
drive.mount('/content/drive')

#Upload Credentials Gmail Ouath
from google.colab import files
files.upload()

# ================================
# Gmail Compliance Assistant (Colab/Jupyter Safe)
# ================================

import os, pickle, base64, json, re, torch
from email.mime.text import MIMEText

# Google Auth
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# LangChain / HuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch, json, re

# -------------------------
# Gmail Authentication (manual flow)
# -------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

def get_gmail_service():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Manual flow: works in Colab/Jupyter
            flow = Flow.from_client_secrets_file(
                'credentials.json',
                scopes=SCOPES,
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'
            )
            auth_url, _ = flow.authorization_url(prompt='consent')
            print("🔗 Please go to this URL and authorize:", auth_url)

            code = input("Paste the authorization code here: ")
            flow.fetch_token(code=code)
            creds = flow.credentials

        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build("gmail", "v1", credentials=creds)

# -------------------------
# ComplianceAssistant (LLM + RAG)
# -------------------------
class ComplianceAssistant:
    def __init__(self):
        # Embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # FAISS Index (adjust path!)
        self.faiss_index = FAISS.load_local(
            "/content/drive/MyDrive/FCA_Project/faiss_index",
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )

        # LLaMA Model
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95
        )

        llm = HuggingFacePipeline(
            pipeline=gen_pipe,
            pipeline_kwargs={"return_full_text": False}
        )

        retriever = self.faiss_index.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 15}
        )

        prompt_text = """
You are an FCA compliance assistant.
You will receive FCA Handbook extracts (COBS 4 only) as CONTEXT and an EMAIL.
Your job is to decide if the EMAIL complies with the CONTEXT.

If you are not sure from the CONTEXT, your decision must be "Insufficient context".

Allowed decisions:
- "Compliant"
- "Not Compliant"
- "Insufficient context"

Rules:
- Use ONLY the CONTEXT; do not rely on outside knowledge.
- Cite the specific COBS 4 sections you used (e.g., "COBS 4.2").
- Rewrite the EMAIL only if your decision is "Not Compliant".
- If decision is "Compliant" or "Insufficient context", the "email" field must be "".
- Keep answers short, professional, and JSON only (no explanations outside JSON).

⚠️ Rewriting rules:
- Preserve ALL factual details from the EMAIL (numbers, percentages, dates, names, descriptors like "low-risk").
- Do NOT invent, add, or paraphrase disclaimers, warnings, or risk statements unless the exact wording appears verbatim in CONTEXT.
- If CONTEXT provides mandatory disclaimer wording verbatim, insert it exactly as written (no changes).
- You must NEVER change descriptors (e.g., do not replace "low-risk" with "high-risk").
- You must NEVER keep absolute guarantee terms like "guaranteed", "guarantee", "no risk", "risk-free", or "assured". Replace them with neutral alternatives such as "offers", "may", or "potential". If no compliant rewrite is possible, remove the offending part entirely.
- Only remove or rephrase wording that is misleading or prohibited by the cited COBS 4 section.
- Keep the rewrite as short and as close as possible to the original EMAIL.

⚠️ Strict formatting rule:
Return ONLY ONE JSON object.
Do not provide multiple alternatives.
Do not repeat decisions.

Return your answer strictly in this format:

<JSON>
{{
  "decision": "Compliant" | "Not Compliant" | "Insufficient context",
  "sections": ["COBS 4.x", ...],
  "email": "Rewritten email if decision is 'Not Compliant', otherwise empty string"
}}
</JSON>

CONTEXT:
{context}

EMAIL:
{question}

⚠️ Output ONLY one JSON object. Do not add explanations, labels, or extra text.
Begin directly with <JSON> and end with </JSON>.
"""
        prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def check_email(self, email_text):
        res = self.qa_chain.invoke({"query": email_text})
        matches = re.findall(r"<JSON>(.*?)</JSON>", res["result"], re.DOTALL)
        if not matches:
            return {"decision": "ParseError", "sections": [], "email": ""}
        try:
            return json.loads(matches[0].strip())
        except json.JSONDecodeError:
            return {"decision": "JSONError", "sections": [], "email": ""}

# -------------------------
# Gmail Helpers
# -------------------------
def fetch_emails(service, label="INBOX", max_results=5):
    emails = []
    if label == "DRAFT":
        drafts = service.users().drafts().list(userId="me", maxResults=max_results).execute()
        if "drafts" in drafts:
            for draft in drafts["drafts"]:
                draft_msg = service.users().drafts().get(userId="me", id=draft["id"]).execute()
                snippet = draft_msg["message"].get("snippet", "")
                emails.append({"id": draft["id"], "snippet": snippet, "source": "DRAFT"})
    else:
        results = service.users().messages().list(userId="me", labelIds=[label], maxResults=max_results).execute()
        if "messages" in results:
            for msg in results["messages"]:
                msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
                snippet = msg_data.get("snippet", "")
                emails.append({"id": msg["id"], "snippet": snippet, "source": label})
    return emails

def get_email_body(service, msg_id, draft=False):
    if draft:
        draft_msg = service.users().drafts().get(userId="me", id=msg_id).execute()
        message = draft_msg["message"]
    else:
        message = service.users().messages().get(userId="me", id=msg_id).execute()

    payload = message.get("payload", {})
    parts = payload.get("parts", [])
    body = ""
    if parts:
        for part in parts:
            if part.get("mimeType") == "text/plain" and "data" in part["body"]:
                body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                break
    return body

def create_draft(service, to, subject, body_text):
    message = MIMEText(body_text)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    draft = service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
    return draft

# -------------------------
# Main Execution Flow
# -------------------------
if __name__ == "__main__":
    service = get_gmail_service()
    assistant = ComplianceAssistant()

    # Choose email source
    source = input("Choose source (INBOX / DRAFT / SENT): ").upper()
    emails = fetch_emails(service, label=source, max_results=5)

    for idx, e in enumerate(emails):
        print(f"{idx}. {e['snippet']}")

    choice = int(input("Select email index: "))
    email_id = emails[choice]["id"]
    body = get_email_body(service, email_id, draft=(emails[choice]["source"]=="DRAFT"))

    print("\n--- ORIGINAL EMAIL ---")
    print(body)

    result = assistant.check_email(body)
    print("\n--- COMPLIANCE RESULT ---")
    print(json.dumps(result, indent=2))

    if result.get("email"):
        action = input("\nSave compliant version as draft? (y/n): ")
        if action.lower() == "y":
            to = input("Recipient: ")
            subject = input("Subject: ")
            draft = create_draft(service, to, subject, result["email"])
            print("✅ Compliant draft saved:", draft["id"])

