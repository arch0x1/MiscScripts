import os
import sys
import json
import psutil
import asyncio
import requests
from xml.etree import ElementTree
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

# Ensure output directory exists
Path(__output__).mkdir(exist_ok=True)

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

@dataclass
class ExtractedKnowledge:
    url: str
    title: str
    raw_content: str
    filtered_content: str
    metadata: Dict[str, Any]
    extracted_at: str
    content_hash: str
    raw_word_count: int
    filtered_word_count: int
    compression_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class KnowledgeProcessor:
    @staticmethod
    def calculate_compression_ratio(raw_length: int, filtered_length: int) -> float:
        if raw_length == 0:
            return 0.0
        return round((1 - filtered_length / raw_length) * 100, 2)
    
    @classmethod
    def process_crawl_result(cls, result, url: str) -> Optional[ExtractedKnowledge]:
        try:
            if not result or not result.success:
                return None
            
            raw_content = result.markdown.raw_markdown if result.markdown else ""
            filtered_content = result.markdown.fit_markdown if result.markdown else ""
            
            if len(filtered_content.strip()) < 100:
                return None
            
            metadata = {
                'status_code': getattr(result, 'status_code', None),
                'response_headers': dict(getattr(result, 'response_headers', {})),
                'load_time': getattr(result, 'load_time', None),
                'success': result.success,
                'screenshot_path': getattr(result, 'screenshot', None),
                'crawl4ai_metadata': getattr(result, 'metadata', {}),
                'links_found': len(getattr(result, 'links', [])),
                'images_found': len(getattr(result, 'images', [])),
            }
            
            content_hash = hashlib.md5(filtered_content.encode('utf-8')).hexdigest()
            
            raw_word_count = len(raw_content.split())
            filtered_word_count = len(filtered_content.split())
            compression_ratio = cls.calculate_compression_ratio(len(raw_content), len(filtered_content))
            
            return ExtractedKnowledge(
                url=url,
                title=getattr(result, 'title', '') or url.split('/')[-1],
                raw_content=raw_content,
                filtered_content=filtered_content,
                metadata=metadata,
                extracted_at=datetime.now().isoformat(),
                content_hash=content_hash,
                raw_word_count=raw_word_count,
                filtered_word_count=filtered_word_count,
                compression_ratio=compression_ratio
            )
            
        except Exception as e:
            print(f"Error processing result for {url}: {e}")
            return None

class KnowledgeStorage:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        (self.output_dir / "individual").mkdir(exist_ok=True)
        (self.output_dir / "consolidated").mkdir(exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.knowledge_items: List[ExtractedKnowledge] = []
        self.seen_hashes: set = set()
    
    def add_knowledge(self, knowledge: ExtractedKnowledge) -> bool:
        if knowledge.content_hash in self.seen_hashes:
            print(f"Skipping duplicate content: {knowledge.url}")
            return False
        
        self.seen_hashes.add(knowledge.content_hash)
        self.knowledge_items.append(knowledge)
        
        self._save_individual_item(knowledge)
        return True
    
    def _save_individual_item(self, knowledge: ExtractedKnowledge):
        import re
        safe_filename = re.sub(r'[^\w\-_.]', '_', knowledge.url.split('//')[-1])[:100]
        
        json_path = self.output_dir / "individual" / f"{safe_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge.to_dict(), f, indent=2, ensure_ascii=False)
        
        md_path = self.output_dir / "individual" / f"{safe_filename}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {knowledge.title}\n\n")
            f.write(f"**Source:** {knowledge.url}\n")
            f.write(f"**Extracted:** {knowledge.extracted_at}\n")
            f.write(f"**Words:** {knowledge.filtered_word_count:,} (filtered from {knowledge.raw_word_count:,})\n")
            f.write(f"**Compression:** {knowledge.compression_ratio}% content removed\n\n")
            f.write("---\n\n")
            f.write(knowledge.filtered_content)
    
    def create_consolidated_dataset(self):
        if not self.knowledge_items:
            print("No knowledge items to consolidate")
            return
        
        sorted_items = sorted(self.knowledge_items, key=lambda x: x.filtered_word_count, reverse=True)
        
        dataset_path = self.output_dir / "consolidated" / f"knowledge_dataset_{self.session_id}.json"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump([item.to_dict() for item in sorted_items], f, indent=2, ensure_ascii=False)
        
        llm_ready_path = self.output_dir / "consolidated" / f"llm_ready_{self.session_id}.txt"
        with open(llm_ready_path, 'w', encoding='utf-8') as f:
            f.write("# Knowledge Base\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Documents: {len(sorted_items)}\n")
            f.write(f"Total Words (filtered): {sum(item.filtered_word_count for item in sorted_items):,}\n")
            f.write(f"Total Words (raw): {sum(item.raw_word_count for item in sorted_items):,}\n\n")
            
            for item in sorted_items:
                f.write(f"\n{'='*100}\n")
                f.write(f"SOURCE: {item.url}\n")
                f.write(f"TITLE: {item.title}\n")
                f.write(f"WORDS: {item.filtered_word_count:,} (compressed {item.compression_ratio}%)\n")
                f.write(f"{'='*100}\n\n")
                f.write(item.filtered_content)
                f.write("\n\n")
        
        print(f"Knowledge base created: {dataset_path}, {llm_ready_path}")

async def crawl_with_extraction(urls: List[str], max_concurrent: int = 5, 
                                content_filter_threshold: float = 0.48,
                                min_word_threshold: int = 100):
    storage = KnowledgeStorage(__output__)
    processor = KnowledgeProcessor()
    
    peak_memory = 0
    process = psutil.Process(os.getpid())
    
    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Memory: {current_mem // (1024 * 1024)} MB (Peak: {peak_memory // (1024 * 1024)} MB)")

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=[
            "--disable-gpu", 
            "--disable-dev-shm-usage", 
            "--no-sandbox",
            "--disable-web-security",
            "--disable-extensions"
        ],
    )
    
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=content_filter_threshold,
                threshold_type="fixed",
                min_word_threshold=min_word_threshold
            )
        )
    )

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            log_memory("Pre-batch")
            
            tasks = []
            for j, url in enumerate(batch):
                session_id = f"session_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append((url, task))

            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (url, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    print(f"Error crawling {url}: {str(result)[:100]}...")
                    continue
                
                if result and result.success:
                    knowledge = processor.process_crawl_result(result, url)
                    if knowledge and storage.add_knowledge(knowledge):
                        print(f"Extracted: {url} ({knowledge.filtered_word_count} words)")
                    else:
                        print(f"Skipped: {url} (insufficient/duplicate content)")
                else:
                    print(f"Failed: {url}")
            
            log_memory("Post-batch")

        storage.create_consolidated_dataset()

    finally:
        await crawler.close()
        log_memory("Final")
        print(f"Peak memory: {peak_memory // (1024 * 1024)} MB")

def get_sitemap_urls(sitemap_url: str) -> List[str]:
    try:
        print(f"Fetching sitemap: {sitemap_url}")
        response = requests.get(sitemap_url, timeout=15, headers={
            'User-Agent': 'Knowledge Crawler 1.0'
        })
        response.raise_for_status()
        
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        print(f"Found {len(urls)} URLs in sitemap")
        return urls
        
    except Exception as e:
        print(f"Sitemap error: {e}")
        return []

async def main():
    sitemap_url = input("Enter sitemap URL: ").strip()
    if not sitemap_url:
        print("No sitemap URL provided. Exiting.")
        return
    
    urls = get_sitemap_urls(sitemap_url)
    
    if not urls:
        print("No URLs found. Exiting.")
        return
    
    await crawl_with_extraction(
        urls, 
        max_concurrent=5,
        content_filter_threshold=0.48,
        min_word_threshold=100
    )
    print("Crawling and extraction complete.")

if __name__ == "__main__":
    asyncio.run(main())
