import json
import os
import time
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import AzureChatOpenAI
from custom_exceptions.exceptions import (
    InvalidQueryError,
    LLMResponseParseError,
    MissingAPIKeyError,
)
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key:
    raise MissingAPIKeyError("AZURE_OPENAI_API_KEY is missing")

response_schemas = [
    ResponseSchema(
        name="entities",
        description="List of entities mentioned, each with a name, a type (e.g., brand, product, food item), and related categories from the provided list.",
    ),
    ResponseSchema(
        name="keywords",
        description="Important action or context words related to the query (e.g., 'want', 'buy', 'find').",
    ),
    ResponseSchema(
        name="references",
        description="Entities used as spatial references (e.g., 'next to Nike').",
    ),
    ResponseSchema(
        name="spatial_phrases",
        description="Location-based phrases (e.g., 'near the entrance', 'on the second floor').",
    ),
    ResponseSchema(
        name="temporal_phrases",
        description="Time-related expressions (e.g., 'today', 'weekend', 'next week').",
    ),
    ResponseSchema(
        name="services",
        description="Relevant services from the provided list if applicable to the query.",
    ),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = PromptTemplate(
    template=(
        "You are a highly intelligent assistant trained to extract structured information from customer queries related to mall services, brands, and products.\n\n"
        "Your task is to analyze the query and extract the following information:\n"
        "- **Entities**: Identify all mentioned entities, including:\n"
        "  - Brand names (e.g., 'Burger King', 'Nike').\n"
        "  - Food items (e.g., 'burger', 'pizza', 'coffee').\n"
        "  - Products (e.g., 'shoes', 'watch', 'handbag').\n"
        "  For each entity, specify its name, type (brand, food item, product), and related categories from the provided {categories} list. If no category applies, use 'Uncategorized'.\n"
        "- **Keywords**: Extract action or context words (e.g., 'want', 'buy', 'find', 'eat').\n"
        "- **References**: Identify entities used as spatial references (e.g., 'next to Nike').\n"
        "- **Spatial Phrases**: Extract location-based phrases (e.g., 'near the entrance', 'on the second floor').\n"
        "- **Temporal Phrases**: Extract time-related expressions (e.g., 'today', 'weekend', 'next week').\n"
        "- **Services**: Select relevant services from the {services} list only if explicitly or implicitly relevant to the query (e.g., 'ATM' for 'withdraw cash').\n\n"
        "Ensure that food items like 'burger' are recognized as entities of type 'food item' and brand names like 'Burger King' are recognized as entities of type 'brand'. If an entity could belong to multiple categories, include all relevant ones.\n\n"
        "{format_instructions}\n\n"
        "Query: {query}\n"
        "Categories: {categories}\n"
        "Services: {services}"
    ),
    input_variables=["query", "categories", "services"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

api_version = "2024-12-01-preview"
llm = AzureChatOpenAI(
    api_version=api_version,
    azure_deployment="gpt-4o-mini",
    temperature=0,
)

chain = prompt | llm | parser


def extract_structured_data(query: str) -> dict:
    """
    Extract structured information from a natural language query using an LLM.

    Args:
        query (str): The user's natural language query string.

    Returns:
        dict: Parsed structured data including entities, keywords, references,
               spatial and temporal phrases, and services.

    Raises:
        InvalidQueryError: If the query is empty or None.
        MissingAPIKeyError: If the OpenAI API key is missing or invalid.
        LLMResponseParseError: If the LLM response cannot be parsed properly.
    """
    if not query or not query.strip():
        raise InvalidQueryError("Query cannot be empty.")

    try:
        start_time = time.time()
        categories = [
            "Women's shoes",
            "Casual & Fine Dining",
            "Lingerie - In home lounge wear, swimwear, hoisery",
            "Watches",
            "Cosmetics",
            "Jewellery",
            "Food items/specialities",
            "Tobacco/news stand",
            "Toddlers specialised area",
            "Fashion accessories",
            "Electronic games",
            "Full range sports wear and equipment",
            "Chicken Cuisine",
            "Beauty Salon Spa Nail Bar",
            "Home improvements",
            "Uncategorized",
            "Handbags-female wallets",
            "Books, newspaper, recorded music, Gifts, Electronics, Media department Store",
            "Fitness centre / spa's",
            "Pharmacy",
            "Men's Arabic - Thobes",
            "Value / Discount Department Stores",
            "Optical",
            "Car rental-Car rental",
            "Coffee & Light Dining",
        ]
        services = [
            "ATM",
            "First Aid Room",
            "Baby Changing Room",
            "Wheelchair Access",
            "Lost & Found",
            "Customer Services",
            "Baby Stroller",
            "Parking",
            "Men's Prayer Room",
            "Women's Prayer Room",
            "Men's Washroom",
            "Women's Washroom",
            "Free Wi-Fi",
        ]

        try:
            result = chain.invoke(
                {
                    "query": query.strip(),
                    "categories": categories,
                    "services": services,
                }
            )
            print(result)
            print(f"Response time: {time.time() - start_time:.2f} seconds")
            if not isinstance(result, dict):
                raise LLMResponseParseError(f"Unexpected LLM output: {result}")
            return result
        except KeyError as e:
            print(e)

    except KeyError as ke:
        raise MissingAPIKeyError(str(ke)) from ke
    except Exception as e:
        raise LLMResponseParseError(original_exception=e) from e