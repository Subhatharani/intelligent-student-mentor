import os
import json
import re
from pathlib import Path
from typing import List, Dict
import PyPDF2
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

NOTES_DIR = "notes"
OUTPUT_FILE = "data/processed_chunks.json"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


class DocumentProcessor:
    
    def __init__(self, notes_dir: str = NOTES_DIR):
        self.notes_dir = Path(notes_dir)
        self.chunks = []
        self.notes_dir.mkdir(exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        print(f" Processing: {pdf_path.name}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += page_text
                
                if len(text.strip()) > 100 and self._is_meaningful_text(text):
                    print(f"    Extracted text directly ({len(text)} chars)")
                    return text
                else:
                    print(f"     Text extraction failed, using OCR...")
                    return self._ocr_pdf(pdf_path)
                    
        except Exception as e:
            print(f"    Error with direct extraction: {e}")
            print(f"    Falling back to OCR...")
            return self._ocr_pdf(pdf_path)
    
    def _is_meaningful_text(self, text: str) -> bool:
        alphanumeric = sum(c.isalnum() for c in text)
        total = len(text)
        return (alphanumeric / total) > 0.6 if total > 0 else False
    
    def _ocr_pdf(self, pdf_path: Path) -> str:
        try:
            print(f"     Converting PDF to images...")
            images = convert_from_path(str(pdf_path), dpi=300)
            
            text = ""
            for i, image in enumerate(images):
                print(f"    OCR on page {i+1}/{len(images)}...")
                page_text = pytesseract.image_to_string(image)
                text += f"\n--- Page {i+1} ---\n{page_text}"
            
            print(f"    OCR complete ({len(text)} chars)")
            return text
            
        except Exception as e:
            print(f"    OCR failed: {e}")
            print(f"    Make sure Tesseract is installed and in your PATH")
            return ""
    
    def chunk_text_smart(self, text: str, filename: str) -> List[Dict]:
        text = self._clean_text(text)
        
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_endings.split(text)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < CHUNK_SIZE:
                current_chunk += sentence + " "
            else:
                if len(current_chunk.strip()) > 100:
                    chunks.append({
                        "id": f"{filename}_{chunk_id}",
                        "text": current_chunk.strip(),
                        "source": filename,
                        "chunk_index": chunk_id,
                        "char_count": len(current_chunk.strip())
                    })
                    chunk_id += 1
                
                overlap = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else ""
                current_chunk = overlap + sentence + " "
        
        if len(current_chunk.strip()) > 100:
            chunks.append({
                "id": f"{filename}_{chunk_id}",
                "text": current_chunk.strip(),
                "source": filename,
                "chunk_index": chunk_id,
                "char_count": len(current_chunk.strip())
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        return text.strip()
    
    def process_all_pdfs(self) -> List[Dict]:
        print("\n Starting document ingestion pipeline...\n")
        
        pdf_files = list(self.notes_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"  No PDF files found in {self.notes_dir}")
            print(f" Add your IT notes (PDFs) to the '{self.notes_dir}' folder and run again.")
            return []
        
        print(f" Found {len(pdf_files)} PDF file(s)\n")
        
        all_chunks = []
        
        for pdf_path in pdf_files:
            try:
                text = self.extract_text_from_pdf(pdf_path)
                
                if not text.strip():
                    print(f"     No text extracted from {pdf_path.name}, skipping...\n")
                    continue
                
                chunks = self.chunk_text_smart(text, pdf_path.stem)
                all_chunks.extend(chunks)
                
                avg_size = sum(c['char_count'] for c in chunks) / len(chunks) if chunks else 0
                print(f"    Created {len(chunks)} chunks (avg size: {avg_size:.0f} chars)\n")
                
            except Exception as e:
                print(f"    Error processing {pdf_path.name}: {e}\n")
                continue
        
        self.chunks = all_chunks
        return all_chunks
    
    def save_chunks(self, output_path: str = OUTPUT_FILE):
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        print(f" Saved {len(self.chunks)} chunks to {output_path}")
    
    def print_summary(self):
        print("\n" + "="*60)
        print(" INGESTION SUMMARY")
        print("="*60)
        print(f"Total chunks created: {len(self.chunks)}")
        
        if self.chunks:
            avg_chunk_size = sum(c['char_count'] for c in self.chunks) / len(self.chunks)
            min_chunk_size = min(c['char_count'] for c in self.chunks)
            max_chunk_size = max(c['char_count'] for c in self.chunks)
            sources = set(c['source'] for c in self.chunks)
            
            print(f"Average chunk size: {avg_chunk_size:.0f} characters")
            print(f"Chunk size range: {min_chunk_size} - {max_chunk_size} characters")
            print(f"Sources processed: {len(sources)}")
            print(f"\nFiles processed:")
            for source in sorted(sources):
                count = sum(1 for c in self.chunks if c['source'] == source)
                avg_size = sum(c['char_count'] for c in self.chunks if c['source'] == source) / count
                print(f"  - {source}.pdf: {count} chunks (avg: {avg_size:.0f} chars)")
            
            print(f"\n Sample chunk preview:")
            sample = self.chunks[0]
            preview_text = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
            print(f"   Source: {sample['source']}")
            print(f"   Length: {sample['char_count']} chars")
            print(f"   Text: {preview_text}")
        else:
            print("  No chunks created. Add PDFs to the 'notes' folder.")
        
        print("="*60 + "\n")


def main():
    processor = DocumentProcessor()
    chunks = processor.process_all_pdfs()
    
    if chunks:
        processor.save_chunks()
    
    processor.print_summary()


if __name__ == "__main__":
    main()