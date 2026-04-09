import re
import string
from typing import List

import nltk
import numpy as np

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Legal-domain stopwords — expanded for Supreme Court language
LEGAL_STOPWORDS = {
    "court", "case", "said", "also", "would", "could", "shall", "may",
    "upon", "within", "thereof", "hereby", "whereas", "aforesaid",
    "aforementioned", "ibid", "viz", "etc", "supra", "infra",
    "plaintiff", "defendant", "petitioner", "respondent", "appellant",
    "learned", "counsel", "honourable", "mr", "ms", "mrs", "justice",
    "present", "dated", "islamabad", "lahore", "karachi", "peshawar",
    "order", "judgment", "appeal", "supreme", "high", "district",
    "bench", "bench-i", "bench-ii", "aor", "asc", "date", "hearing",
    "versus", "para", "page", "pp", "vol", "ref",
}


class TextPreprocessor:
    """
    NLP text preprocessing pipeline for legal documents.
    Handles long Supreme Court judgment text (up to 4000 chars).

    Samsung curriculum:
      - OOP: class with __init__, multiple methods
      - NLP: cleaning, tokenization, stopword removal, lemmatization
      - Python: regex, list comprehensions, type hints
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_word_length: int = 2,
        language: str = "english",
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize        = lemmatize
        self.min_word_length  = min_word_length

        self._stop_words  = set(stopwords.words(language)) | LEGAL_STOPWORDS
        self._lemmatizer  = WordNetLemmatizer()

        # Compiled regex patterns
        self._re_html        = re.compile(r"<[^>]+>")
        self._re_url         = re.compile(r"https?://\S+|www\.\S+")
        self._re_email       = re.compile(r"\S+@\S+")
        self._re_section_ref = re.compile(
            r"\b(section|sec|article|art|clause|order|rule)\s*\d+[\w\-]*",
            re.IGNORECASE,
        )
        self._re_whitespace  = re.compile(r"\s+")
        self._re_nonalpha    = re.compile(r"[^a-z0-9\s]")
        # Remove Supreme Court header boilerplate
        self._re_header      = re.compile(
            r"(supreme court of pakistan|appellate jurisdiction|present:|"
            r"mr\. justice|date of hearing|for the appellant|for the respondent)",
            re.IGNORECASE,
        )

    def clean(self, text: str) -> str:
        """Basic text cleaning pipeline."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = self._re_html.sub(" ", text)
        text = self._re_url.sub(" ", text)
        text = self._re_email.sub(" ", text)

        # Preserve section references as single tokens
        def preserve_section(m):
            return m.group(0).replace(" ", "_").replace("-", "_")
        text = self._re_section_ref.sub(preserve_section, text)

        text = text.translate(str.maketrans("", "", string.punctuation))
        text = self._re_nonalpha.sub(" ", text)
        text = self._re_whitespace.sub(" ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """NLTK word tokenization."""
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Filter stopwords and short tokens."""
        return [
            t for t in tokens
            if t not in self._stop_words and len(t) >= self.min_word_length
        ]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Reduce words to base form."""
        return [self._lemmatizer.lemmatize(t) for t in tokens]

    def process(self, text: str) -> str:
        """Full preprocessing pipeline → single cleaned string."""
        text   = self.clean(text)
        tokens = self.tokenize(text)
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        return " ".join(tokens)

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a list of texts — used for training data prep."""
        return [self.process(t) for t in texts]

    def extract_legal_keywords(self, text: str) -> dict:
        """
        Extract structured legal info from raw case text.
        Works on real Supreme Court judgment language.
        """
        text_lower = text.lower()

        # Extract statute references
        section_pattern = re.compile(
            r"(section|sec\.?|article|art\.?|rule|order)\s*(\d+[\w\-]*(?:\s*[,&]\s*\d+[\w\-]*)*)",
            re.IGNORECASE,
        )
        sections = list({m.group(0).strip() for m in section_pattern.finditer(text)})

        # Case type detection
        case_type_keywords = {
            "criminal":       ["murder", "302", "theft", "robbery", "fir", "prosecution",
                               "penal", "ppc", "cnsa", "peca", "terrorism", "kidnapping"],
            "family":         ["divorce", "khul", "custody", "maintenance", "nikah", "marriage",
                               "talaq", "guardian", "minor", "family court"],
            "civil":          ["contract", "breach", "damages", "injunction", "property",
                               "land", "compensation", "tort", "civil appeal", "suit"],
            "corporate":      ["company", "shareholder", "director", "secp", "arbitration",
                               "securities", "winding up"],
            "constitutional": ["fundamental rights", "article 25", "article 199", "article 184",
                               "writ", "habeas corpus", "constitutional petition"],
            "service":        ["civil servant", "dismissal from service", "service tribunal",
                               "compulsory retirement", "government employee"],
            "tax":            ["income tax", "sales tax", "fbr", "customs", "tax appeal"],
        }

        detected_types = [
            ct for ct, kws in case_type_keywords.items()
            if any(kw in text_lower for kw in kws)
        ]

        # Jurisdiction detection
        jurisdiction_keywords = {
            "sindh":       ["karachi", "sindh", "shc", "hyderabad"],
            "punjab":      ["lahore", "punjab", "lhc", "faisalabad"],
            "kpk":         ["peshawar", "kpk", "khyber", "mardan"],
            "balochistan": ["quetta", "balochistan"],
            "federal":     ["islamabad", "federal", "ihc", "supreme court", "nab"],
        }
        detected_jurs = [
            j for j, kws in jurisdiction_keywords.items()
            if any(kw in text_lower for kw in kws)
        ]

        # Top keywords by frequency
        from collections import Counter
        tokens   = self.tokenize(self.clean(text))
        filtered = self.remove_stop_words(tokens)
        freq     = Counter(filtered)
        top_kw   = [w for w, _ in freq.most_common(10)]

        return {
            "sections":      sections[:10],
            "case_types":    detected_types,
            "jurisdictions": detected_jurs if detected_jurs else ["federal"],
            "keywords":      top_kw,
        }


if __name__ == "__main__":
    pp = TextPreprocessor()
    sample = (
        "SUPREME COURT OF PAKISTAN. The accused was charged under Section 302 PPC "
        "for the murder of the deceased in Karachi. The Sindh High Court convicted him. "
        "Defense counsel argues self-defense under Section 304 PPC."
    )
    print("Original:", sample)
    print("Processed:", pp.process(sample))
    print("Keywords:", pp.extract_legal_keywords(sample))
