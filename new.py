import sys
if sys.platform.startswith("win"):
    import asyncio
    def silence_event_loop_closed_error(*args, **kwargs):
        pass
    if hasattr(asyncio, "proactor_events"):
        try:
            from asyncio.proactor_events import _ProactorBasePipeTransport
            _ProactorBasePipeTransport.__del__ = silence_event_loop_closed_error
        except Exception:
            pass

import os
import sys
import asyncio
import aiohttp
import logging
import json
import time
import argparse
import random
import io
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Core libraries
import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Environment variables for API keys
from dotenv import load_dotenv



# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TableData:
    """Class to hold table data and metadata"""
    dataframe: pd.DataFrame
    table_index: int
    column_names: List[str]
    row_count: int
    source_url: str

@dataclass
class LLMConfig:
    """Configuration for LLM APIs"""
    name: str
    api_type: str  # 'huggingface' or 'gemini'
    model_name: str
    api_key: str
    endpoint: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7

class WebScraper:
    """Hybrid scraper combining anti-bot measures with reliable table detection"""
    
    def __init__(self, timeout: int = 30):
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36...'
        ]
        self.timeout = timeout
        self._setup_session()

    def _setup_session(self):
        """Configure headers with random user agent rotation"""
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml...',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1'
        }
        self.session.headers.update(headers)

    def scrape_url(self, url: str, max_retries: int = 3) -> Tuple[Optional[BeautifulSoup], List[TableData]]:
        """Dual-mode scraping with fallback logic"""
        soup = None  # Ensure soup is always defined
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Primary method: Direct URL access with headers
                tables = self._pandas_direct_extract(url)
                
                # Fallback 1: HTML content parsing
                if not tables:
                    tables = self._pandas_html_extract(response.text, url)
                
                # Fallback 2: BeautifulSoup extraction
                if not tables:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    tables = self._bs4_extract(soup, url)
                else:
                    soup = BeautifulSoup(response.content, 'html.parser')

                return soup, tables

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(random.uniform(2, 5))
        # If all attempts fail, return None and empty list
        return soup, []

    def _pandas_direct_extract(self, url: str) -> List[TableData]:
        """Original reliable pandas method with header injection"""
        try:
            with requests.Session() as pandas_session:
                pandas_session.headers.update(self.session.headers)
                response = pandas_session.get(url)
                dfs = pd.read_html(
                    io.BytesIO(response.content), 
                    header=0,
                    encoding=response.encoding
                )
                table_data_list = []
                for i, df in enumerate(dfs):
                    if df.shape[0] == 0:
                        logger.warning(f"[pandas_direct_extract] Table {i} is empty and will be skipped.")
                        continue
                    logger.info(f"[pandas_direct_extract] Table {i} shape: {df.shape}")
                    table_data_list.append(TableData(
                        dataframe=df,
                        table_index=i,
                        column_names=df.columns.tolist(),
                        row_count=len(df),
                        source_url=url
                    ))
                return table_data_list
        except Exception as e:
            logger.warning(f"Direct pandas extraction failed: {e}")
            return []
        
    def _pandas_html_extract(self, html: str, url: str) -> List[TableData]:
        """Enhanced HTML content parser"""
        try:
            dfs = pd.read_html(
                io.StringIO(html),
                header=0,
                encoding='utf-8',
                attrs={'class': None}  # Remove restrictive class filtering
            )
            table_data_list = []
            for i, df in enumerate(dfs):
                if df.shape[0] == 0:
                    logger.warning(f"[pandas_html_extract] Table {i} is empty and will be skipped.")
                    continue
                logger.info(f"[pandas_html_extract] Table {i} shape: {df.shape}")
                table_data_list.append(TableData(
                    dataframe=df,
                    table_index=i,
                    column_names=df.columns.tolist(),
                    row_count=len(df),
                    source_url=url
                ))
            return table_data_list
        except Exception as e:
            logger.warning(f"HTML content extraction failed: {e}")
            return []

    def _bs4_extract(self, soup: BeautifulSoup, url: str) -> List[TableData]:
        """Robust fallback parser with column normalization"""
        tables = []
        for i, table in enumerate(soup.find_all('table')):
            try:
                rows = [
                    [cell.get_text(strip=True) for cell in row.find_all(['td','th'])]
                    for row in table.find_all('tr')
                ]
                if not rows or len(rows) < 2:
                    logger.warning(f"[bs4_extract] Table {i} is empty or has only headers and will be skipped.")
                    continue
                df = pd.DataFrame(rows)
                df.columns = df.iloc[0]  # Use first row as headers
                df = df[1:]  # Remove header row
                logger.info(f"[bs4_extract] Table {i} shape: {df.shape}")
                tables.append(TableData(df, i, df.columns.tolist(), len(df), url))
            except Exception as e:
                logger.error(f"Table {i} extraction failed: {e}")
        return tables

class LLMManager:
    """Enhanced LLM manager with current working models"""

    def __init__(self, llm_configs: List[LLMConfig]):
        self.llm_configs = llm_configs
        self.session = None

    async def initialize(self):
        """Initialize aiohttp session"""
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()

    async def generate_content_for_table(self, table_data: TableData) -> Dict[str, List[str]]:
        """Generate content for all rows of a table using all configured LLMs"""
        if not self.session:
            await self.initialize()

        results = {}

        for llm_config in self.llm_configs:
            try:
                logger.info(f"Generating content using {llm_config.name}")
                paragraphs = await self._generate_paragraphs_for_llm(table_data, llm_config)
                results[llm_config.name] = paragraphs
            except Exception as e:
                logger.error(f"Error with {llm_config.name}: {e}")
                results[llm_config.name] = [f"Error generating content: {e}"] * table_data.row_count

        return results

    async def _generate_paragraphs_for_llm(self, table_data: TableData, llm_config: LLMConfig) -> List[str]:
        """Generate paragraphs for all rows using a specific LLM"""
        tasks = []

        for index, row in table_data.dataframe.iterrows():
            task = self._generate_paragraph_for_row(row, table_data.column_names, llm_config)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        paragraphs = []
        for result in results:
            if isinstance(result, Exception):
                paragraphs.append(f"Error generating content: {result}")
            else:
                paragraphs.append(result)

        return paragraphs

    async def _generate_paragraph_for_row(self, row: pd.Series, columns: List[str], llm_config: LLMConfig) -> str:
        """Generate a paragraph for a single row using the specified LLM"""
        try:
            # Create a prompt based on the row data
            row_data = []
            for col in columns:
                value = row.get(col, '')
                if value:
                    row_data.append(f"{col}: {value}")

            row_text = ", ".join(row_data)

            prompt = f"""
            Based on the following data from a table row, write a comprehensive and informative paragraph that explains the information in a natural, flowing manner. Make the content engaging and provide context where appropriate.

            Data: {row_text}

            Write a well-structured paragraph (3-5 sentences) that presents this information in a coherent and readable format:
            """

            if llm_config.api_type == 'huggingface':
                return await self._call_huggingface_api(prompt, llm_config)
            elif llm_config.api_type == 'gemini':
                return await self._call_gemini_api(prompt, llm_config)
            else:
                raise ValueError(f"Unsupported API type: {llm_config.api_type}")

        except Exception as e:
            logger.error(f"Error generating paragraph: {e}")
            return f"Error generating content for this row: {e}"

    async def _call_huggingface_api(self, prompt: str, config: LLMConfig) -> str:
        """Make API call to Hugging Face with current working models"""
        try:
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }

            endpoint = config.endpoint or f"https://api-inference.huggingface.co/models/{config.model_name}"

            # Enhanced payload for better text generation
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "return_full_text": False,
                    "do_sample": True,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1
                },
                "options": {
                    "wait_for_model": True
                }
            }

            async with self.session.post(endpoint, headers=headers, json=payload) as response:
                if response.status == 503:
                    # Model is loading, wait and retry
                    logger.info(f"Model {config.model_name} is loading, waiting...")
                    await asyncio.sleep(20)
                    async with self.session.post(endpoint, headers=headers, json=payload) as retry_response:
                        retry_response.raise_for_status()
                        result = await retry_response.json()
                else:
                    response.raise_for_status()
                    result = await response.json()

                if isinstance(result, list) and result:
                    return result[0].get('generated_text', '').strip()
                else:
                    return result.get('generated_text', '').strip()

        except Exception as e:
            logger.error(f"Hugging Face API error: {e}")
            raise

    async def _call_gemini_api(self, prompt: str, config: LLMConfig) -> str:
        """Make API call to Google Gemini"""
        try:
            headers = {
                "Content-Type": "application/json"
            }

            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model_name}:generateContent?key={config.api_key}"

            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": config.temperature,
                    "maxOutputTokens": config.max_tokens
                }
            }

            async with self.session.post(endpoint, headers=headers, json=payload) as response:
                response.raise_for_status()
                result = await response.json()

                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content.strip()
                else:
                    return "No content generated"

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

class PDFGenerator:
    """Enhanced PDF generator"""

    def __init__(self, output_dir: str = "output_pdfs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.styles = getSampleStyleSheet()

        # Create custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
        )

        self.table_header_style = ParagraphStyle(
            'TableHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
        )

    def generate_pdf(self, url: str, tables_data: List[TableData], 
                    llm_results: Dict[str, Dict[int, List[str]]]) -> List[str]:
        """Generate PDF files for each LLM"""
        generated_files = []

        for llm_name in llm_results.keys():
            try:
                filename = f"{llm_name}_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                filepath = self.output_dir / filename

                self._create_pdf_for_llm(url, tables_data, llm_results[llm_name], str(filepath), llm_name)
                generated_files.append(str(filepath))
                logger.info(f"Generated PDF: {filepath}")

            except Exception as e:
                logger.error(f"Error generating PDF for {llm_name}: {e}")

        return generated_files

    def _create_pdf_for_llm(self, url: str, tables_data: List[TableData], 
                           llm_content: Dict[int, List[str]], filepath: str, llm_name: str):
        """Create a PDF for a specific LLM"""
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []

        
        # Source URL
        story.append(Paragraph(f"<b>Source URL:</b> {url}", self.styles['Normal']))
        story.append(Spacer(1, 12))

        # Content for each table
        for table_data in tables_data:
            table_idx = table_data.table_index


            # Generated paragraphs for this table
            if table_idx in llm_content:
                for i, paragraph in enumerate(llm_content[table_idx]):
                    story.append(Paragraph(paragraph, self.styles['Normal']))
                    story.append(Spacer(1, 12))

            story.append(PageBreak())

        doc.build(story)

class WebTableScraper:
    """Main orchestrator class"""

    def __init__(self, llm_configs: List[LLMConfig]):
        self.scraper = WebScraper()
        self.llm_manager = LLMManager(llm_configs)
        self.pdf_generator = PDFGenerator()

    async def process_url(self, url: str) -> List[str]:
        """Main method to process a URL and generate PDFs"""
        try:
            # Step 1: Scrape the URL and extract tables
            logger.info("Starting web scraping...")
            soup, tables_data = self.scraper.scrape_url(url)

            if not tables_data:
                logger.warning("No tables found on the webpage")
                return []

            # Step 2: Generate content using LLMs
            logger.info("Generating content using LLMs...")
            all_llm_results = {}

            for table_data in tables_data:
                llm_results = await self.llm_manager.generate_content_for_table(table_data)

                for llm_name, paragraphs in llm_results.items():
                    if llm_name not in all_llm_results:
                        all_llm_results[llm_name] = {}
                    all_llm_results[llm_name][table_data.table_index] = paragraphs

            # Step 3: Generate PDFs
            logger.info("Generating PDF documents...")
            generated_files = self.pdf_generator.generate_pdf(url, tables_data, all_llm_results)

            return generated_files

        except Exception as e:
            logger.error(f"Error processing URL: {e}")
            raise
        finally:
            await self.llm_manager.close()

def setup_llm_configs() -> List[LLMConfig]:
    """Setup LLM configurations with current working models"""
    configs = []

    # Hugging Face configurations with current working models
    hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
    if hf_api_key:
        # Updated working models for 2025
        hf_models = [
            'microsoft/Phi-3.5-mini-instruct',      # Microsoft's latest small model
            'microsoft/Phi-3-mini-4k-instruct',     # Reliable Microsoft model
        ]

        for model in hf_models:
            config = LLMConfig(
                name=f"HuggingFace_{model.split('/')[-1]}",
                api_type='huggingface',
                model_name=model,
                api_key=hf_api_key,
                max_tokens=300,
                temperature=0.7
            )
            configs.append(config)

    # Google Gemini configuration
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if gemini_api_key:
        config = LLMConfig(
            name="Google_Gemini",
            api_type='gemini',
            model_name='gemini-1.5-flash',
            api_key=gemini_api_key,
            max_tokens=400,
            temperature=0.7
        )
        configs.append(config)

    if not configs:
        logger.warning("No API keys found. Please set HUGGINGFACE_API_KEY and/or GEMINI_API_KEY environment variables")

    return configs

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Web Table Scraper and LLM Document Generator")
    parser.add_argument("url", help="URL to scrape tables from")
    parser.add_argument("--output-dir", default="output_pdfs", help="Output directory for PDFs")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts for failed requests")

    args = parser.parse_args()

    # Setup LLM configurations
    llm_configs = setup_llm_configs()

    if not llm_configs:
        logger.error("No LLM configurations available. Please check your API keys.")
        return 1

    # Create scraper instance
    scraper = WebTableScraper(llm_configs)

    try:
        # Process the URL
        generated_files = await scraper.process_url(args.url)

        if generated_files:
            logger.info("Process completed successfully!")
            logger.info("Generated files:")
            for file_path in generated_files:
                logger.info(f"  - {file_path}")
        else:
            logger.warning("No files were generated")

    except Exception as e:
        logger.error(f"Process failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        import sys
        if sys.platform.startswith("win"):
            import asyncio
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass