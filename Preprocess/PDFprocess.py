import re
import fitz
import matplotlib.pyplot as plt
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
import unicodedata
import spacy
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from langchain.schema import Document
factory= IndicNormalizerFactory()
bn_normalizer = factory.get_normalizer("bn")
class PDFprocessor:
  def __init__(self,pdf_path:str):
    self.pdf_path=pdf_path
    self.docs: list[Document] = []
    self.chunks=[]
  
  def load_pdf(self) -> list[Document]:
        """Extract real Unicode text page‑by‑page with PyMuPDF."""
        pdf = fitz.open(self.pdf_path)
        pages = []
        for page in pdf:
            text = page.get_text("text")  # reliable Unicode output
            pages.append(Document(page_content=text, metadata={"page": page.number+1}))
        pdf.close()
        self.docs = pages
        return self.docs
  
  @staticmethod
  def clean_hyphentation(text:str)-> str:
    return re.sub(r"-\n", "", text)
  
  @staticmethod
  def remove_header_footer(text:str)-> str:
    lines=[]
    for line in text.splitlines():
      if re.match(r"^\s*Page\s+\d+\s*$", line):
        continue
      lines.append(line)
    return "\n".join(lines)
  @staticmethod
  def merge_line(text:str):   ## Merge lines that are not separated by a newline
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)
  @staticmethod
  def normalize_quotes(text: str)-> str:

    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return text
  @staticmethod
  def normalize_whitespace(text: str)-> str:
     text=re.sub(r"[ \t]{2,}"," ",text)
     text=re.sub(r"\n{3,}","\n\n",text)
     return text.strip()
  
  def clean(self):
    """Clean the loaded PDF documents."""
    for doc in self.docs:
      text = doc.page_content
      for func in(
          self.clean_hyphentation,
          self.remove_header_footer,
          self.merge_line,
          self.normalize_quotes,
          self.normalize_whitespace
      ):
       text = func(text)
      doc.page_content = text
    return self.docs
  @staticmethod
  def clean_bangla_text(text: str) -> str:
        # Unicode NFC normalization
        text = unicodedata.normalize("NFC", text)

        # strip ASCII + digits
        text = re.sub(r"[A-Za-z0-9\u09E6-\u09EF]", "", text)

        # keep only Bangla block (including diacritics) and whitespace
        text = re.sub(r"[^\u0980-\u09FF\u09BC-\u09C4\u09CD\s।]", "", text)

        # collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Indic orthographic normalization
        text = bn_normalizer.normalize(text)

        # word-tokenize to fix merged/ split errors
        tokens = trivial_tokenize(text, lang="bn")
        return " ".join(tokens)
  def clean_bangla(self, split_sentences: bool = False):
        for doc in self.docs:
            # your regex‐based stripping
            cleaned = self.clean_bangla_text(doc.page_content)

            # Normalize orthography
            cleaned = bn_normalizer.normalize(cleaned)

            # 2) (Optional) Sentence split with Indic NLP
            if split_sentences:
                sents = sentence_tokenize.sentence_split(cleaned, lang="bn")
                # strip out any empties and rejoin with danda if you like
                doc.page_content = [s.strip() for s in sents if s.strip()]
            else:
                # 3) (Optional) Word tokenize to fix merged words
                tokens = indic_tokenize.trivial_tokenize(cleaned, lang="bn")
                # rejoin into a single clean string
                doc.page_content = " ".join(tokens)

        return self.docs
  def get_paragraphs(
          self,
          min_words: int = 5
      ) -> list[str]:
          """
          Return only “true” paragraphs, dropping headers,
          footers, MCQs, option‑lines, and very short fragments.
          """
          paras = []
          for doc in self.docs:
              for raw_para in doc.page_content.split("\n\n"):
                  p = raw_para.strip()
                  if not p:
                      continue

                  # 1) Drop pure page‑number lines
                  if re.fullmatch(r"\s*Page\s*\d+\s*", p):
                      continue

                  # 2) Drop MCQ question lines (e.g. "১. …", "২) …", etc.)
                  if re.match(r"^[০১২৩৪৫৬৭৮৯]+[\.|\)].+", p):
                      continue

                  # 3) Drop MCQ option lines (e.g. "ক) …", "খ) …")
                  if re.match(r"^[কখগঘঙচছজঞ]+[\.|\)].+", p):
                      continue

                  # 4) Drop lines that are just punctuation or too short
                  words = p.split()
                  if len(words) < min_words:
                      continue

                  # 5) (Optionally) Drop lines ending with a question‑mark
                  #    uncomment if you only want descriptive paras:
                  # if p.endswith("?"):
                  #     continue

                  paras.append(p)
          return paras
  def visualize_paragraph_lengths(self, bins: int = 30, figsize: tuple = (8, 4)):
        """
        Plot a histogram of paragraph word‐counts.
        """
        paras = self.get_paragraphs()
        lengths = [len(p.split()) for p in paras]
        plt.figure(figsize=figsize)
        plt.hist(lengths, bins=bins)
        plt.title("Paragraph Length Distribution")
        plt.xlabel("Words per paragraph")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
  #def chunk(self, chunk_size: int = 800, chunk_overlap: int = 100) -> list:
      # coerce any list-based page_content back into a single string
      #for doc in self.docs:
          #if isinstance(doc.page_content, list):
              # join with danda or a space (whichever makes more sense)
              #doc.page_content = "। ".join(doc.page_content)

      #splitter = SentenceParserTextSplitter(
      #chunk_size=chunk_size,
      #chunk_overlap=chunk_overlap,
      #parser="spacy",   # or "nltk"
      #model="xx_ent_wiki_sm"
      #)
      #self.chunks = splitter.split_documents(self.docs)
      #return self.chunks
  def chunk(self, chunk_size: int = 800, chunk_overlap: int = 100) -> list:
    # 1) Flatten list-based page_content
    for doc in self.docs:
        if isinstance(doc.page_content, list):
            doc.page_content = "। ".join(doc.page_content)

    # 2) Load spaCy & add the sentencizer
    try:
        nlp = spacy.load("xx_ent_wiki_sm")
    except OSError:
        raise OSError("Run: python -m spacy download xx_ent_wiki_sm")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    # 3) Sentence-split each page
    sent_docs = []
    for doc in self.docs:
        for sent in nlp(doc.page_content).sents:
            sent_docs.append(
                doc.__class__(page_content=sent.text, metadata=doc.metadata)
            )

    # 4) Character-based re-chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    self.chunks = splitter.split_documents(sent_docs)
    return self.chunks
  def visualize_chunk_lengths(self, bins: int = 30, figsize: tuple = (8,4)):
        """
        Plot a histogram of chunk lengths (in characters).
        Call this after you’ve done `proc.chunk()`.
        """
        if not self.chunks:
            raise ValueError("No chunks found – run `proc.chunk()` first.")
        lengths = [len(c.page_content) for c in self.chunks]
        plt.figure(figsize=figsize)
        plt.hist(lengths, bins=bins)
        plt.title("Chunk Length Distribution (chars)")
        plt.xlabel("Characters per chunk")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
