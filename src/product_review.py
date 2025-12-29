from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import openai

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever


# Response schema for social media sentiment
response_schemas_1 = [
    ResponseSchema(name="Social media sentiment", description='3 bullet points summarizing major highlights about this product'),
    ResponseSchema(name="Sentiment scores (%)", description='Positive-score|Negative-score|Neutral-score')   
]
output_parser_1 = StructuredOutputParser.from_response_schemas(response_schemas_1)
format_instructions_1 = output_parser_1.get_format_instructions()

# Response schema for amazon review sentiment
response_schemas_2 = [
    ResponseSchema(name='Amazon review sentiment', description='3 bullet points summarizing major highlights about this product'),
    ResponseSchema(name="Sentiment scores (%)", description='Positive-score|Negative-score|Neutral-score'),    
]
output_parser_2 = StructuredOutputParser.from_response_schemas(response_schemas_2)
format_instructions_2 = output_parser_2.get_format_instructions()

# Final recommendation schema
response_schemas_3 = [
    ResponseSchema(name="Recommendation", description='3 detailed recommendations for minimalist, balanced and maximalist preferences')]

output_parser_3 = StructuredOutputParser.from_response_schemas(response_schemas_3)
format_instructions_3 = output_parser_3.get_format_instructions()

class ProductReview:
    def __init__(self, query, sub_query,
                 amzn_reviews, amzn_prod_details, sm_data,
                 format_instructions_1, format_instructions_2, format_instructions_3, 
                 openai_key, 
                 output_parser_1, output_parser_2, output_parser_3,
                 state = ["social_media", "amazon", "final"]
                ):
        
        self.query = query
        self.sub_query = sub_query
        self.state = state
        self.format_instructions_1 = format_instructions_1
        self.format_instructions_2 = format_instructions_2
        self.format_instructions_3 = format_instructions_3
        self.openai_key = openai_key
        self.output_parser_1 = output_parser_1
        self.output_parser_2 = output_parser_2
        self.output_parser_3 = output_parser_3
        self.sm_data = sm_data
        self.amzn_reviews = amzn_reviews
        self.amzn_prod_details = amzn_prod_details

    def _create_search_index(self, full_text):
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(full_text)
    
        # creating searchable index
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_key)
        print("Embeddings vector created..")
        return FAISS.from_texts(chunks, embeddings)   
    
    def _analyze_text(self, relevant_text, state, ground_query):
        #client = openai.OpenAI(api_key=openai_key)
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=self.openai_key  # Pass key directly or use environment variable
        )
        if state == "social_media":
            # initiate llm model
            prompt = ChatPromptTemplate.from_template(
            """
            You are a product sentiment analysis assistant. A user is researching products related to this enquiry "{query}" .
            
            Your task:
            1. Analyze the social media corpus below.
            2. Summarize **3 grounded, specific observations** expressed by users related to product performance, experience, or satisfaction.
            3. Provide sentiment scores in the format: "Positive|Negative|Neutral" (no % symbol).
            
            Return the output strictly as valid JSON, with keys:
            - "Social media sentiment": list of 3 bullet points
            - "Sentiment scores (%)": string in format "##|##|##"
            
            Example:
            {{
              "Social media sentiment": [
                "- Gamers appreciate the high frame rates and thermal management",
                "- Several users report inconsistent GPU driver updates",
                "- Users say build quality feels premium and durable"
              ],
              "Sentiment scores (%)": "65(+)|20(-)|15"
            }}
            
            Output only valid JSON. Do not include commentary or formatting hints.
            
            Corpus:
            \"\"\"
            {text}
            \"\"\"
            """)
            
            print("Modelling Social Media Sentiment...")
            fmt = self.format_instructions_1
            parser = self.output_parser_1
        
        elif state == "amazon":
           prompt = ChatPromptTemplate.from_template(
            """
            You are a product sentiment analysis assistant. A user is researching products related to this enquiry "{query}".
            
            Your task:
            1. Analyze verified **Amazon product reviews**.
            2. Summarize 3 **key insights**, focused on product **performance, quality, satisfaction, or drawbacks** relevant to the query.
            3. Provide sentiment scores in the format: "Positive|Negative|Neutral" (no % symbol).
            
            Return the output strictly as valid JSON with:
            - "Amazon review sentiment": list of 3 grounded bullet points
            - "Sentiment scores (%)": string like "##|##|##"
            
            Example:
            {{
              "Amazon review sentiment": [
                "- Customers report the laptop handles AAA games smoothly at high settings",
                "- Complaints about overheating after prolonged gaming sessions",
                "- Battery life praised, lasting 6–8 hours during mixed usage"
              ],
              "Sentiment scores (%)": "70(+)|20(-)|10"
            }}
            
            Only output a valid JSON object.
            
            Corpus:
            \"\"\"
            {text}
            \"\"\"
            """)

           print("Modelling Amazon Review Sentiment...")
    
           fmt = self.format_instructions_2
           parser = self.output_parser_2
            
        
        elif state == "final":
           prompt = ChatPromptTemplate.from_template(
           """
            You are a product recommendation assistant. Return ONLY a valid JSON object.
        
            Task:
            Analyze the product corpus below (regarding "{query}") and provide a recommendation based on the user’s need: "{initial_query}".
        
            Requirements:
            - Extract the most relevant products from the corpus.
            - Justify each product choice using specific advantages or drawbacks mentioned in the corpus.
            - Tailor each recommendation to suit the user's intent (e.g., gaming, travel, durability, etc.).
            - Avoid repeating exact review lines — synthesize meaningful insights.
            - Limit recommendations to 2–3 options max, with 1–2 concise, informative sentences each.
        
            Output Format:
            {{
                "Recommendation": [
                    "- [Product Name] is suitable because [brief explanation based on corpus].",
                    "- [Optional Alternative Product] is also a good fit due to [reason]."
                ]
            }}
        
            Output MUST match the format exactly and be parseable by `json.loads()`.
        
            Corpus:
            \"\"\"
            {text}
            \"\"\"
            """
           )
            
           print("Modelling Final Recommendation...")
           fmt = self.format_instructions_3
           parser = self.output_parser_3
        
        else:
            print("Error: Enter valid stage staus")
        
        messages = prompt.format_messages(
                query = self.sub_query,
                initial_query = ground_query,
                text = relevant_text,
                format_instructions=fmt
            )
        
        response = llm(messages)
        raw_output = response.content
        cleaned_output = re.sub(r'```json|```', '', raw_output).strip()
        print("Result Generation Completed \nResult:\nNow Parsing...")

        try:
            parsed_result = parser.parse(raw_output)
            print("Parsing completed")
            print("--------------------------------------------------------------------------------------------------------------------\n")
            return parsed_result
        except Exception as e:
            print("Failed to parse:", cleaned_output)
            raise
            
    def _analyze_with_semantic_search(self, full_texts, state, ground_query=None):
        
        #full_texts = " ".join(full_texts) if isinstance(full_texts, list) else full_texts
    
        # retreiving relevant chunks
        #using lightweight BM25 pre-filter
        bm25_retriever = BM25Retriever.from_texts(full_texts)
        top_k_docs = bm25_retriever.get_relevant_documents(self.query, k=1000)
        reduced_corpus = [doc.page_content for doc in top_k_docs]
        print("--------------------------------------------------------------------------------------------------------------------")
        print("First layer BM25 Retriever Corpus Reduction Complete")
        # creating vector index on full corpus
        reduced_text = " ".join(reduced_corpus)
        index = self._create_search_index(reduced_text)
        
        faiss_retriever = index.as_retriever()
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], 
            weights=[0.3,0.7]
        )
        
        relevant_docs = ensemble_retriever.get_relevant_documents(self.sub_query)
        relevant_text = " ".join([doc.page_content for doc in relevant_docs])
        print("Semantic Search Completed")
        print(f"Total number of relevant text: {len(relevant_text)}\n")
        
        return self._analyze_text(relevant_text, state, ground_query)

    def run_models(self):
        result_1 = self._analyze_with_semantic_search(self.sm_data, state=self.state[0])
        result_2 = self._analyze_with_semantic_search(self.amzn_reviews, state=self.state[1])
    
        sentiment_summary = {**result_1, **result_2}

        # Transforming into a textual query
        sentiment_query = " ".join([f"{k}: {v}" for k, v in sentiment_summary.items()])

        # Temporarily override self.query to use summary as query
        old_query = self.sub_query
        self.sub_query = sentiment_query
        result_3 = self._analyze_with_semantic_search(self.amzn_prod_details, state=self.state[2], ground_query=old_query)
        self.sub_query = old_query
        
        final_result = {**sentiment_summary, **result_3}
    
        return final_result
        print("--------------------------------------------------------------------------------------------------------------------")

    def evaluate_model(self, ground_truth=None):
        results = self.run_models()

        assert "Social media sentiment" in results
        assert "Amazon review sentiment" in results
        assert "Recommendation" in results

        if ground_truth:
            from sklearn.model_selection import classification_report

            pred_labels = self._convert_sentiment_to_labels(results)
            true_labels = ground_truth

            return classification_report(pred_labels, true_labels)
        
        return "Evaluation passed basic checks"    